# Import only what exists for now
try:
    from .generate import generate_text
except ImportError:
    generate_text = None

try:
    from .kv_cache import KVCache
except ImportError:
    KVCache = None

__all__ = [name for name in ['generate_text', 'KVCache'] if globals()[name] is not None]