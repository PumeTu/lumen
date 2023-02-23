import torch
import torch.nn as nn
from lumen.layers.conv import DepthWiseSeperableConv, ConvBnAct

class PANet(nn.Module):
    """Path Aggregation Network
    """
    def __init__(self, inputs: tuple = ('c3', 'c4', 'c5'), in_channels: tuple = (256, 512, 1024), depthwise: bool = False, activation: str = 'silu'):
        super().__init__()
        self.inputs = inputs
        self.in_channels = in_channels
        self.conv = DepthWiseSeperableConv if depthwise else ConvBnAct 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        
    def forward(self, x):
        raise NotImplementedError

class YOLOv4Head(nn.Module):
    """YOLOv4 Anchor based Head
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _decode_bbox_to_xywh(bbox_pred, anchor):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class YOLOP(nn.Module):
    """YOLO for Panoptic Driving Perception"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError   