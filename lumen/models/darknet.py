import torch
import torch.nn as nn

from typing import List, Tuple

from .layers.conv import ConvBnAct

class DarkResidualBlock(nn.Module):
    """Darknet Residual Block
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        reduced_channels = in_channels // 2
        self.conv1 = ConvBnAct(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0, norm_layer='bn2d', activation='lrelu')
        self.conv2 = ConvBnAct(reduced_channels, in_channels, kernel_size=3, stride=1, padding='same', norm_layer='bn2d', activation='lrelu')

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += shortcut
        return x

class Darknet53(nn.Module):
    def __init__(self, in_channels, stem_out_channels: int = 32, num_blocks: List = [2, 8, 8, 4], output: Tuple = ('c3', 'c4', 'c5')):
        super().__init__()
        self.output = output
        self.stem = self.build_stem_layer(in_channels, stem_out_channels)
        self.c1 = nn.Sequential(*self.build_stage_layer(stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2
        self.c2 = nn.Sequential(*self.build_stage_layer(in_channels, num_blocks=num_blocks[0], stride=2))
        in_channels *= 2
        self.c3 = nn.Sequential(*self.build_stage_layer(in_channels, num_blocks=num_blocks[1], stride=2))
        in_channels *= 2
        self.c4 = nn.Sequential(*self.build_stage_layer(in_channels, num_blocks=num_blocks[2], stride=2))
        in_channels *= 2
        self.c5 = nn.Sequential(*self.build_stage_layer(in_channels, num_blocks=num_blocks[3], stride=2))

    @staticmethod
    def build_stem_layer(in_channels: int, stem_out_channels: int):
        return ConvBnAct(in_channels, stem_out_channels, kernel_size=3, stride=1, padding='same', norm_layer='bn2d', activation='lrelu')
        
    @staticmethod
    def build_stage_layer(in_channels: int, num_blocks: int, stride: int):
        return [
            ConvBnAct(in_channels, in_channels*2, kernel_size=3, stride=stride, padding='same', norm_layer='bn2d', activation='lrelu'),
            *[(DarkResidualBlock(in_channels*2)) for _ in range(num_blocks)]
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

