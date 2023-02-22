import math
from typing import List, Tuple
import torch.nn.functional as F

def get_padding(kernel_size: int, stride: int = 1, dilation:int = 1, **_) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    height, width = x.size()[-2:]
    pad_height, pad_width = get_same_padding(height, k[0], s[0], d[0]), get_same_padding(width, k[1], s[1], d[1])
    if pad_height > 0 or pad_width > 0:
        x = F.pad(x, [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2], value=value)
    return x

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple:
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            padding = get_padding(kernel_size, **kwargs)
    return padding

