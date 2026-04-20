"""Campaign-MEM experiment package."""

# On Windows CUDA environments, importing NumPy/scikit-learn before PyTorch can
# break DLL resolution for torch. Import torch eagerly at package load time so
# downstream modules inherit the safe order.
import torch  # noqa: F401

from .utils import load_yaml, save_json, set_seed

__all__ = ["load_yaml", "save_json", "set_seed"]
