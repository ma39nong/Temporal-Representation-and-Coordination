"""Read numeric LeRobot v2.1 episodes without loading videos or OpenPI."""
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LeRobotAdapter:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if not isinstance(self.info, dict):
            raise ValueError('Dataset info must be an object')
        if self.info.get('codebase_version') != 'v2.1':
            raise ValueError('Only explicitly verified LeRobot v2.1 supported')
        fps = self.info.get('fps')
        import math
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid dataset FPS')
        self.episodes = read_jsonl(self.root / 'meta/episodes.jsonl')
        if any(not isinstance(r, dict) for r in self.episodes):
            raise ValueError('Episode metadata rows must be objects')
        ids = [r.get('episode_index') for r in self.episodes]
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or duplicate episode IDs')
        if any(type(r.get('length')) is not int or r['length'] <= 0 for r in self.episodes):
            raise ValueError('Invalid episode lengths')
        if len(ids) != self.info.get('total_episodes') or sum(r['length'] for r in self.episodes) != self.info.get('total_frames'):
            raise ValueError('Episode metadata totals mismatch')

    def episode_path(self, episode_index):
        chunk_size = self.info.get('chunks_size', 1000)
        if type(episode_index) is not int or episode_index < 0 or type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError('Invalid episode index or chunks_size')
        path = (self.root / self.info['data_path'].format(
            episode_index=episode_index, episode_chunk=episode_index // chunk_size)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Data template escapes dataset root')
        return path

    def read_episode(self, episode_index, *, extra_columns=()):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('Parquet audit requires pip install -e ".[parquet]"') from exc
        file = pq.ParquetFile(self.episode_path(episode_index))
        desired = ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index',
                   'source_timestamp_ns', 'observation.arm_engaged', 'observation.engaged']
        columns = list(dict.fromkeys(k for k in [*desired, *extra_columns] if k in file.schema_arrow.names))
        table = file.read(columns=columns)
        # Preserve fixed-list Arrow child float32; to_pylist would obscure source dtype.
        import numpy as np
        result = {}
        for key in columns:
            col = table[key].combine_chunks()
            if key in ('action', 'observation.state'):
                rows = col.to_pylist()
                try:
                    # An empty child slice preserves the Arrow numeric dtype without
                    # invoking the optional pandas type-conversion shim.
                    dtype = col.values.slice(0, 0).to_numpy(zero_copy_only=False).dtype
                    result[key] = np.asarray(rows, dtype=dtype)
                except (AttributeError, TypeError, ValueError):
                    result[key] = np.asarray(rows, dtype=object)
            else:
                result[key] = col.to_numpy(zero_copy_only=False)
        return result
