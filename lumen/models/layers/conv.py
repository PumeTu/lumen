import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Optional, Tuple

from padding import pad_same, get_padding_value

def create_conv2d_pad(in_channels, out_channels, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

def create_conv(in_channels: int, out_channels: int, kernel_size: Union[int, tuple] = 1, **kwargs):
    padding = kwargs['padding']
    if padding == 'same':
        return create_conv2d_pad(in_channels, out_channels, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)

def get_activation(name: str = "relu", inplace: bool = True):
    """ Get activation function 
    Args:
        name (str): name of activation function
        inplace (bool): specify whether to the operation inplace or not
    Returns:
        module (nn.Module): activation function
    """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f"Unsupported activation function: {name}")
    return module

def get_norm_layer(name: str, num_features: int):
    """ Get normalization layer
    Args:
        name (str): name of normalization layer
        num_feature (int): number of input channels=
    Returns:
        module (nn.Module): normalization layer
    """
    if name == "bn2d":
        module = nn.BatchNorm2d(num_features=num_features, eps=0.001, momentum=0.03)
    else:
        raise AttributeError(f"Unsupported normalization layer: {name}")
    return module


class ConvBnAct(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, tuple] = 1,
                stride: Optional[Union[int, tuple]] = 1,
                padding: Optional[Union[int, tuple, str]] = '',
                dilation: Optional[Union[int, tuple]] = 1,
                groups: Optional[int] = 1,
                bias: Optional[bool] = False,
                norm_layer: str = 'Batch',
                activation: str = 'relu',
                inplace: bool = True):
        super().__init__()
        self.conv = create_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = get_norm_layer(norm_layer)
        self.act = get_activation(activation, inplace)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

def conv2d_same(x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                stride: Tuple[int, int] = (1, 1), 
                padding: Tuple[int, int] = (0, 0),
                dilation: Tuple[int, int] = (1, 1),
                groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation, groups)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ SAME convlution wrapper for 2D convolutions
    """ 
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, tuple] = 1,
                stride: Optional[Union[int, tuple]] = 1,
                padding: Optional[Union[int, tuple, str]] = 0,
                dilation: Optional[Union[int, tuple]] = 1,
                groups: Optional[int] = 1,
                bias: Optional[bool] = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)