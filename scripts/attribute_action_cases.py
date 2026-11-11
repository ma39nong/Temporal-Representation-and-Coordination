"""Targeted CPU attribution of the existing 12 review entries, no model calls.

Reads only the nine implicated episodes and existing frozen training predictions.
Local source/proof roots must be supplied explicitly in a private JSON config.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from trc.analysis.flash.crs import selected_npz_rows
from trc.analysis.flash.metrics import EVAL_GROUPS, summary
from trc.data.action_schema import ORDER, canonicalize, verify_features
from trc.data.qualification import inverse_compact_time, flash_window_cuts


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    resolve = lambda key: (config_path.parent / cfg[key]).resolve()
    export, out = resolve('export'), resolve('output')
    out.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = json.loads((export / 'metadata.json').read_text())
    cases = json.loads(resolve('cases').read_text())
    manifest = json.loads(resolve('manifest').read_text())
    records = {r['source_recording_id']: r for r in manifest['records']}
    wanted = {c['source_recording_id'] for c in cases}
    ix = [i for i, w in enumerate(meta['windows']) if w['source_recording_id'] in wanted and w['training_source'] is True]
    values = {key: selected_npz_rows(export/'aligned.npz', key, ix)
              for key in ('truth', 'oracle', 'prediction', 'history', 'valid')}
    if not values['valid'].all():
        raise ValueError('Fully valid frozen windows required')
    remote = json.loads(resolve('remote_proofs').read_text())
    raw_inventory = json.loads(resolve('raw_bag_inventory').read_text())
    if len(cases) != 12 or len(wanted) != 9 or len(ix) != 45:
        raise ValueError('This entry point audits the frozen 12-entry / nine-source case set only')
    provenance, episode_data, rows = {}, {}, []
    for source in sorted(wanted):
        record = records[source]
        name = Path(record['root']).name
        root = Path(cfg['roots'][name])
        ep = record['episode']
        local_info = json.loads((root/'meta/info.json').read_text())
        verify_features(local_info['features'], 'converter58')
        epmeta = next(json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()
                      if json.loads(line)['episode_index'] == ep)
        if epmeta['source_recording_id'] != source or epmeta['length'] != record['rows']:
            raise ValueError('Snapshot source identity/length differs')
        p = root / record['info']['data_path'].format(episode_chunk=ep//1000, episode_index=ep)
        names = pq.read_schema(p).names
        t = pq.read_table(p, columns=[k for k in ('action', 'observation.state', 'timestamp', 'frame_index',
                                                 'source_timestamp_ns', 'audit.original_timestamp_ns') if k in names])
        a = canonicalize(np.asarray(t['action'].to_pylist(), np.float32), 'converter58')
        s = canonicalize(np.asarray(t['observation.state'].to_pylist(), np.float32), 'converter58', state=True)
        np.testing.assert_array_equal(np.asarray(t['frame_index']), np.arange(len(a)))
        np.testing.assert_allclose(np.asarray(t['timestamp']), np.arange(len(a))/30, atol=1e-4, rtol=0)
        if name == '10.16before':
            proof_path = Path(cfg['legacy_proof_root'])/f'{ep:06d}.json'
            proof = json.loads(proof_path.read_text())
            proof_hash = sha(proof_path)
        else:
            proof = next(v for v in remote if v['episode'] == ep and v['dataset'] in name)
            proof_path = Path(proof['path']); proof_hash = proof['sha256']
        proof_ep = next(v for v in proof['episodes'] if v['episode_index'] == ep)
        for key in ('source_start_ns', 'source_end_ns'):
            if int(proof_ep[key]) != int(epmeta[key]):
                raise ValueError('Proof interval not bound to selected snapshot')
        if proof_ep['length'] != len(a):
            raise ValueError('Proof length differs')
        raw, cuts = inverse_compact_time(np.asarray(t['source_timestamp_ns']), proof['cleaning']['mapping'])
        original_difference = None
        if 'audit.original_timestamp_ns' in t.column_names:
            original_difference = int(np.max(np.abs(raw-np.asarray(t['audit.original_timestamp_ns']))))
            if original_difference:
                raise ValueError('Original audit clock disagrees with deletion map')
        if name == '10.16before':
            raw_match = next(v for v in raw_inventory if v['group'] == '10.16之前数据'
                             and v['episode'] == int(source.split('episode')[-1]))
            old_hashes = proof['production']['original']['sha256']
            if (raw_match['metadata_sha256'] != old_hashes['metadata.yaml']
                    or raw_match['state_sha256'] != old_hashes['collection_state.json']):
                raise ValueError('Legacy raw bag alias metadata differs from archived original')
            raw_binding = 'original collection_state and metadata SHA match archived production proof'
        else:
            raw_match = next(v for v in raw_inventory if v['source_recording_id'] == source)
            raw_binding = 'UUID matches conversion episode and current bag collection_state'
        if not raw_match['metadata_exists'] or not raw_match['bag_files']:
            raise ValueError('Missing raw bag metadata or volumes')
        provenance[source] = {'snapshot': str(root), 'parquet_sha256': sha(p),
                              'available_raw_bag': raw_match, 'raw_binding': raw_binding, 'proof_path': str(proof_path),
                              'proof_sha256': proof_hash, 'raw_source': proof['cleaning']['source'],
                              'cuts': cuts, 'original_audit_max_difference_ns': original_difference,
                              'episode': ep, 'rows': len(a), 'source_start_ns': epmeta['source_start_ns']}
        episode_data[source] = (root, a, s, raw)
    for j, index in enumerate(ix):
        w = meta['windows'][index]; source = w['source_recording_id']; start = w['current_frame']
        root, a, s, raw = episode_data[source]
        np.testing.assert_array_equal(values['truth'][j], a[start:start+50])
        np.testing.assert_array_equal(values['history'][j], s[start-7:start+1])
        e = values['prediction'][j].astype(float)-values['truth'][j]
        ro = values['oracle'][j].astype(float)-values['truth'][j]
        pe = values['prediction'][j].astype(float)-values['oracle'][j]
        groups = {}
        for group, gidx in EVAL_GROUPS.items():
            ge = e[:, gidx]; re = ro[:, gidx]; pred = pe[:, gidx]
            tick, local_joint = np.unravel_index(np.abs(ge).argmax(), ge.shape)
            rt, rj = np.unravel_index(np.abs(re).argmax(), re.shape)
            pt, pj = np.unravel_index(np.abs(pred).argmax(), pred.shape)
            groups[group] = {'representation_worst_tick': int(rt), 'representation_worst_joint': ORDER[gidx[rj]],
                             'prediction_worst_tick': int(pt), 'prediction_worst_joint': ORDER[gidx[pj]],
                             'total': summary(ge), 'representation': summary(re), 'prediction': summary(pred),
                             'last9_total': summary(ge[-9:]), 'last9_oracle': summary(re[-9:]),
                             'last9_prediction': summary(pred[-9:]), 'first41_total': summary(ge[:41]),
                             'increment': summary(np.diff(ge, axis=0)),
                             'worst_tick': int(tick), 'worst_joint': ORDER[gidx[local_joint]],
                             'mse_cross_term': float(2*np.mean(re*pred))}
        rows.append({'export_window': index, 'source_recording_id': source, 'frame': start,
                     'crossing_cuts': flash_window_cuts(start, provenance[source]['cuts']),
                     'window_support': [start-7, start+51], 'raw_current_ns': int(raw[start]),
                     'aligned_truth_history_exact': True, 'groups': groups,
                     'gt_increment_rms': summary(np.diff(values['truth'][j], axis=0))['rmse']})
    entries = []
    for n, case in enumerate(cases):
        high = next(r for r in rows if r['export_window'] == case['export_window'])
        # Same recording, disjoint full support, verified continuity. Match motion
        # magnitude within factor two when possible; phase remains unverified.
        eligible = [r for r in rows if r['source_recording_id'] == high['source_recording_id']
                    and not r['crossing_cuts'] and abs(r['frame']-high['frame']) > 58
                    and r['groups']['full_action']['total']['rmse'] < high['groups']['full_action']['total']['rmse']]
        matched = [r for r in eligible if .5 <= r['gt_increment_rms']/max(high['gt_increment_rms'], 1e-12) <= 2]
        pool = matched or eligible
        control = min(pool, key=lambda r: r['groups']['full_action']['total']['rmse']) if pool else None
        entries.append({'entry': n+1, 'selection_metric': case['selection_metric'], 'high': high,
                        'control': control, 'control_motion_matched_factor2': bool(matched),
                        'phase_match_status': 'needs_video_review', 'coordination_failure': 'not_established'})
    (out/'case_attribution.json').write_text(json.dumps({'entries': entries, 'provenance': provenance,
            'all_45_source_windows': rows}, indent=2))
    import csv
    table = []
    for entry in entries:
        for kind in ('high', 'control'):
            row = entry[kind]
            if row is None:
                continue
            for group in ('left_arm', 'left_hand', 'right_arm', 'right_hand'):
                g = row['groups'][group]
                table.append({'entry': entry['entry'], 'kind': kind, 'export_window': row['export_window'],
                    'source_recording_id': row['source_recording_id'], 'frame': row['frame'],
                    'group': group, 'crossing_cuts': str(row['crossing_cuts']),
                    'total_rmse': g['total']['rmse'], 'oracle_rmse': g['representation']['rmse'],
                    'prediction_oracle_rmse': g['prediction']['rmse'], 'worst_joint': g['worst_joint'],
                    'worst_tick': g['worst_tick'], 'last9_rmse': g['last9_total']['rmse']})
    with (out/'paired_groups.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    from trc.analysis.flash.diagnosis import coefficient_contributions
    focus = [next(e for e in entries if e['entry'] == n) for n in (7, 8, 9)]
    fi = [ix.index(e['high']['export_window']) for e in focus]
    coefficients = [selected_npz_rows(export/'aligned.npz', k, [ix[i] for i in fi]) for k in ('c_gt','c_pred')]
    with np.load(export/'aligned.npz') as z:
        coeff_report = coefficient_contributions(values['truth'][fi], values['oracle'][fi], values['prediction'][fi],
                 *coefficients, z['scale'], z['basis'], z['q01'], z['q99'])
    (out/'focus_coefficients.json').write_text(json.dumps(coeff_report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for line, entry in enumerate(focus):
        for col, kind in enumerate(('high', 'control')):
            row = entry[kind]; j = ix.index(row['export_window']); ax = axes[line, col]
            for key, label in (('truth','GT'),('oracle','Oracle'),('prediction','Prediction')):
                ax.plot(np.arange(50)/30, values[key][j, :, 6], label=label)
            ax.axvspan(41/30,49/30,alpha=.12,color='red',label='last 9 ticks' if line == 0 else None)
            ax.set_title(f"entry {entry['entry']} {kind}: frame {row['frame']} / left_fr3_joint7")
            ax.set_xlabel('Future time (s)'); ax.set_ylabel('rad')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('REAL training-source windows; clean support; controls not certified phase-matched')
    fig.tight_layout(); fig.savefig(out/'left_arm_paired.png',dpi=160); plt.close(fig)

    # Small, accurately sought, six-frame contact sheets from each implicated
    # camera/window. No full-video copy/transcode and no raw DB3 reads.
    frame_offsets = [0, 7, 17, 32, 47, 56]
    media = []
    unique = {r['export_window']: r for en in entries for r in (en['high'], en['control']) if r is not None}
    for index, row in (unique.items() if cfg.get('render_media', True) else []):
        source = row['source_recording_id']; root, _, _, _ = episode_data[source]
        rec = records[source]; directory = out/'contacts'/str(index); directory.mkdir(parents=True, exist_ok=True)
        canvas = Image.new('RGB', (6*224, 3*188), 'white'); draw = ImageDraw.Draw(canvas)
        videos = []
        for cam in range(3):
            path = root / rec['info']['video_path'].format(episode_chunk=rec['episode']//1000,
                    episode_index=rec['episode'], video_key=f'observation.images.cam{cam}')
            video = {'path': str(path), 'exists': path.is_file()}
            if not path.is_file():
                raise FileNotFoundError(path)
            video['sha256'] = sha(path)
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=nb_frames,avg_frame_rate,start_time', '-of', 'json', str(path)], text=True)
            video['stream'] = json.loads(probe)['streams'][0]
            if int(video['stream']['nb_frames']) != rec['rows'] or video['stream']['avg_frame_rate'] != '30/1':
                raise ValueError('Video frame count / frequency differs')
            filt = '+'.join(f'eq(n\\,{v})' for v in frame_offsets)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1',
                    '-ss', str((row['frame']-7)/30), '-i', str(path), '-an', '-vf', f'select={filt},scale=224:160',
                    '-frames:v', '6', '-fps_mode', 'vfr', '-y', str(directory/f'cam{cam}_%02d.png')], check=True)
            for k, off in enumerate(frame_offsets):
                frame = directory/f'cam{cam}_{k+1:02d}.png'
                with Image.open(frame) as im:
                    canvas.paste(im.convert('RGB'), (k*224, cam*188+28))
                draw.text((k*224+3,cam*188+3), f'cam{cam} frame {row["frame"]-7+off} t={(off-7)/30:.3f}', fill='black')
            videos.append(video)
        canvas.save(directory/'sheet.jpg', quality=90)
        media.append({'export_window':index, 'source_recording_id':source, 'frame':row['frame'],
                      'sheet':str(directory/'sheet.jpg'), 'videos':videos})
        print(json.dumps({'contact_window': index, 'crossing': row['crossing_cuts']}), flush=True)
    if media:
        (out/'media.json').write_text(json.dumps(media, indent=2))
    identity = {'data_kind': 'real', 'scope': 'selected training-source case attribution only',
                'metadata_sha256':sha(export/'metadata.json'), 'aligned_sha256':sha(export/'aligned.npz'),
                'config_sha256':sha(config_path), 'script_sha256':sha(__file__),
                'checkpoint_parameter_sha256':meta['checkpoint_parameter_sha256'],
                'normalization_sha256':meta['normalization_sha256'],
                'code_dirty': bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                'raw_inventory_sha256': sha(resolve('raw_bag_inventory')),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'model_forwards':0,'training_updates':0,'bag_messages_read':0,
                'entries':len(entries), 'unique_case_windows':len({en['high']['export_window'] for en in entries}),
                'crossing_entries':sum(bool(en['high']['crossing_cuts']) for en in entries)}
    (out/'identity.json').write_text(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
