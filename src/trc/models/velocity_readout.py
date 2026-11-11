"""Array-only residual math and strict parameter extension, usable without JAX."""
import numpy as np

MODES = ('none', 'unified', 'order_specific')
ADAPTER_PATHS = {('action_out_proj', 'adapter_down'), ('action_out_proj', 'adapter_up')}


def adapter_shapes(mode, width=1024, action_dim=58, orders=7):
    if mode not in MODES or (width, action_dim, orders) != (1024, 58, 7):
        raise ValueError('Pinned FLASH width1024, 58D, seven orders required')
    if mode == 'none':
        return {}
    rank = 11 if mode == 'unified' else 8
    return {'adapter_down': (width, rank),
            'adapter_up': (rank, action_dim) if mode == 'unified' else (orders, rank, action_dim)}


def initialize_adapter(mode, seed=42003):
    shapes = adapter_shapes(mode)
    if not shapes:
        return {}
    rng = np.random.default_rng(seed)
    return {'adapter_down': (rng.standard_normal(shapes['adapter_down']) / np.sqrt(1024)).astype(np.float32),
            'adapter_up': np.zeros(shapes['adapter_up'], np.float32)}


def adapter_count(mode):
    return sum(int(np.prod(shape)) for shape in adapter_shapes(mode).values())


def residual(hidden, down, up, mode, *, xp=np):
    """Seven simultaneous coefficient tokens, never a time/frequency partition."""
    if mode not in ('unified', 'order_specific') or hidden.ndim != 3 or hidden.shape[1:] != (7,1024):
        raise ValueError('Expected (B,7,1024) readout hidden')
    shapes = adapter_shapes(mode)
    if down.shape != shapes['adapter_down'] or up.shape != shapes['adapter_up']:
        raise ValueError('Adapter parameter layout differs')
    z = xp.einsum('bkh,hr->bkr', hidden, down)
    return xp.einsum('bkr,rd->bkd' if mode == 'unified' else 'bkr,krd->bkd', z, up)


def strict_extend_flat(reference, loaded, mode, seed=42003):
    """Extend complete original flat parameters; no missing old leaf allowed.

    reference includes new adapter shapes; loaded is the ORIGINAL checkpoint.
    Resumed augmented checkpoints use exact config.load, never this initializer.
    """
    adapter = {('action_out_proj', k): v for k,v in initialize_adapter(mode,seed).items()}
    if set(reference)-set(adapter) != set(loaded) or set(adapter)-set(reference):
        raise ValueError('Missing/unexpected original checkpoint parameter paths')
    for path, value in loaded.items():
        if value.shape != reference[path].shape or value.dtype != reference[path].dtype:
            raise ValueError('Original parameter shape/dtype differs: '+str(path))
    for path, value in adapter.items():
        if value.shape != reference[path].shape or value.dtype != reference[path].dtype:
            raise ValueError('Adapter shape/dtype differs')
    return {**loaded, **adapter}
