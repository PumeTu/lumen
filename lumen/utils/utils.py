import torch
import numpy as np

from functools import partial

def multi_apply(fn, *args, **kwargs):
    """Apply function to a list of arguments
    Note:
        This function applies the 'fn' to multiple inputs and map the outputs of the 'fn' into different list.
        Each list contains the same type of outputs corresponding to different outputs
    Args:
        fn (function): function that will be applied to a list of arguments
    Returns:
        tuple(list): tuple containing multiple list, each list with the returned results of the function
    """
    pfn = partial(fn, **kwargs) if kwargs else fn
    results = map(pfn, *args)
    return tuple(map(list, zip(*results)))