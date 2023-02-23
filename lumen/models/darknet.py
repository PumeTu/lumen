import torch
import torch.nn as nn

from typing import List, Tuple

from lumen.layers.conv import ConvBnAct, DepthWiseSeperableConv

class DarknetBottleneck(nn.Module):
    """Darknet Residual Block
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels: int, depthwise: bool = False, activation: str = 'silu'):
        super().__init__()
        self.conv = DepthWiseSeperableConv if depthwise else ConvBnAct
        reduced_channels = in_channels // 2
        self.conv1 = self.conv(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0, norm_layer='bn2d', activation=activation)
        self.conv2 = self.conv(reduced_channels, in_channels, kernel_size=3, stride=1, padding='same', norm_layer='bn2d', activation=activation)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += shortcut
        return x

class Darknet53(nn.Module):
    """Darknet53
    """
    def __init__(self, in_channels, stem_out_channels: int = 32, num_blocks: List = [2, 8, 8, 4], output: Tuple = ('c3', 'c4', 'c5')):
        super().__init__()
        self.output = output
        self.stem = self._build_stem_layer(in_channels, stem_out_channels)
        self.c1 = nn.Sequential(*self._build_stage_layer(stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2
        self.c2 = nn.Sequential(*self._build_stage_layer(in_channels, num_blocks=num_blocks[0], stride=2))
        in_channels *= 2
        self.c3 = nn.Sequential(*self._build_stage_layer(in_channels, num_blocks=num_blocks[1], stride=2))
        in_channels *= 2
        self.c4 = nn.Sequential(*self._build_stage_layer(in_channels, num_blocks=num_blocks[2], stride=2))
        in_channels *= 2
        self.c5 = nn.Sequential(*self._build_stage_layer(in_channels, num_blocks=num_blocks[3], stride=2))

    @staticmethod
    def _build_stem_layer(in_channels: int, stem_out_channels: int):
        return ConvBnAct(in_channels, stem_out_channels, kernel_size=3, stride=1, padding='same', norm_layer='bn2d', activation='lrelu')
        
    @staticmethod
    def _build_stage_layer(in_channels: int, num_blocks: int, stride: int):
        return [
            ConvBnAct(in_channels, in_channels*2, kernel_size=3, stride=stride, padding='same', norm_layer='bn2d', activation='lrelu'),
            *[(DarknetBottleneck(in_channels*2)) for _ in range(num_blocks)]
        ]

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        x = self.c1(x)
        outputs['c1'] = x
        x = self.c2(x)
        outputs['c2'] = x
        x = self.c3(x)
        outputs['c3'] = x
        x = self.c4(x)
        outputs['c4'] = x
        x = self.c5(x)
        outputs['c5'] = x
        return {k:v for k, v in outputs.items() if k in self.output}

class Focus(nn.Module):
    """YOLOv5 Focus layer to reduce layers, parameters FLOPS, and CUDA memory increases forward and backward 
        speed while minimally impacting mAP, replaces stem layer from normal Darknet53
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: str = 1, norm_layer: str = 'bn2d', activation: str = 'silu'):
        super().__init__()
        self.conv = ConvBnAct(in_channels*4, out_channels, kernel_size=kernel_size, stride=stride, norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class CSPLayer(nn.Module):
    """CSP Bottleneck with 3 Convolutions
    """
    def __init__(self, in_channels: int, out_channels: int, n: int, expansion=0.5, depthwise: bool = False, activation: str = 'silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBnAct(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation)
        self.conv2 = ConvBnAct(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation)
        self.conv3 = ConvBnAct(2*hidden_channels, out_channels, kernel_size=1, stride=1, activation=activation)
        self.bottleneck = nn.Sequential(*[DarknetBottleneck(hidden_channels, depthwise, activation) for _ in range(n)])

    def forward(self, x):
        x_ = self.conv1(x)
        x = self.conv2(x)
        x = self.bottleneck(x)
        x = torch.cat((x, x_), dim=1)
        return self.conv3(x)


class CSPDarknet53(nn.Module):
    """Cross Stage Partial Darknet 53
    """
    def __init__(self, in_channels: int, width_multiplyer: float = 1.0, depth_multiplyer: float = 1.0, depthwise: bool = False, output: Tuple = ('c3', 'c4', 'c5'), activation: str = 'silu'):
        super().__init__()
        self.output = output
        self.conv = DepthWiseSeperableConv if depthwise else ConvBnAct
        base_channels = int(width_multiplyer * 64)
        base_depth = max(round(depth_multiplyer*3), 1)
        self.stem = self._build_stem_layer(in_channels, base_channels)
        self.c2 = self._build_stage_layer(base_channels, base_channels*2, base_depth, depthwise=depthwise, activation=activation)
        self.c3 = self._build_stage_layer(base_channels*2, base_channels*4, base_depth*3, depthwise=depthwise, activation=activation)
        self.c4 = self._build_stage_layer(base_channels*4, base_channels*8, base_depth*3, depthwise=depthwise, activation=activation)
        self.c5 = self._build_stage_layer(base_channels*8, base_channels*16, base_depth, depthwise=depthwise, activation=activation)

    @staticmethod
    def _build_stem_layer(in_channels, base_channels, kernel_size: int = 3, stride: int = 1, norm_layer: str = 'bn2d', activation: str = 'silu'):
        return Focus(in_channels, base_channels, kernel_size, stride, norm_layer, activation)

    def _build_stage_layer(self, in_channels: int, out_channels: int, n: int, depthwise: bool, activation: str = 'silu'):
        return nn.Sequential(
            self.conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation),
            CSPLayer(out_channels, out_channels, n, depthwise=depthwise, activation=activation)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        x = self.c2(x)
        outputs['c2'] = x
        x = self.c3(x)
        outputs['c3'] = x
        x = self.c4(x)
        outputs['c4'] = x
        x = self.c5(x)
        outputs['c5'] = x
        return {k:v for k, v in outputs.items() if k in self.output}