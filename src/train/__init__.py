# Import only what exists for now
try:
    from .dataset import TextDataset
except ImportError:
    TextDataset = None

try:
    from .eval_utils import evaluate_model
except ImportError:
    evaluate_model = None

try:
    from .utils import get_optimizer, get_scheduler
except ImportError:
    get_optimizer = None
    get_scheduler = None

try:
    from .trainer import train_model, train
except ImportError:
    train_model = None
    train = None

__all__ = [name for name in ['train_model', 'train', 'TextDataset', 'evaluate_model', 'get_optimizer', 'get_scheduler'] if globals()[name] is not None]