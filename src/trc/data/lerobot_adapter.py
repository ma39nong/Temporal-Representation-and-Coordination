"""Read numeric LeRobot v2.1 episodes without loading videos or OpenPI."""
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LeRobotAdapter:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if self.info.get('codebase_version') != 'v2.1':
            raise ValueError('Only explicitly verified LeRobot v2.1 supported')
        self.episodes = read_jsonl(self.root / 'meta/episodes.jsonl')

    def episode_path(self, episode_index):
        chunk_size = self.info.get('chunks_size', 1000)
        path = (self.root / self.info['data_path'].format(
            episode_index=episode_index, episode_chunk=episode_index // chunk_size)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Data template escapes dataset root')
        return path

    def read_episode(self, episode_index):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('Parquet audit requires pip install -e ".[parquet]"') from exc
        file = pq.ParquetFile(self.episode_path(episode_index))
        desired = ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index',
                   'source_timestamp_ns', 'observation.arm_engaged', 'observation.engaged']
        columns = [k for k in desired if k in file.schema_arrow.names]
        table = file.read(columns=columns)
        # Preserve fixed-list Arrow child float32; to_pylist would obscure source dtype.
        import numpy as np
        result = {}
        for key in columns:
            col = table[key].combine_chunks()
            if key in ('action', 'observation.state'):
                rows = col.to_pylist()
                try:
                    result[key] = np.asarray(rows, dtype=col.type.value_type.to_pandas_dtype())
                except (TypeError, ValueError):
                    result[key] = np.asarray(rows, dtype=object)
            else:
                result[key] = col.to_numpy(zero_copy_only=False)
        return result
