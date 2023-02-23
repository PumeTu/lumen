import torch
import torch.nn as nn

from typing import List

from lumen.layers.conv import DepthWiseSeperableConv, ConvBnAct
from .darknet import CSPLayer
from lumen.utils.utils import multi_apply

class PANet(nn.Module):
    """Path Aggregation Network
    """
    def __init__(self, inputs: tuple = ('c3', 'c4', 'c5'), in_channels: tuple = (256, 512, 1024), depthwise: bool = False, activation: str = 'silu'):
        super().__init__()
        self.inputs = inputs
        self.in_channels = in_channels
        self.conv = DepthWiseSeperableConv if depthwise else ConvBnAct 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.lateral_conv = ConvBnAct(in_channels[2], in_channels[1], kernel_size=1, stride=1, padding='same', activation=activation)
        self.CSP_p4 = CSPLayer(in_channels[1]*2, in_channels[1], n=3, depthwise=depthwise, activation=activation)
        self.topdown_conv = ConvBnAct(in_channels[1], in_channels[0], kernel_size=1, stride=1, padding='same', activation=activation)
        self.CSP_p3 = CSPLayer(in_channels[0]*2, in_channels[0], n=3, depthwise=depthwise, activation=activation)
        self.bottomup_conv1 = ConvBnAct(in_channels[0], in_channels[0], kernel_size=3, stride=2, activation=activation)
        self.CSP_n3 = CSPLayer(in_channels[0], in_channels[1], n=3, depthwise=depthwise, activation=activation)
        self.bottomup_conv2 = ConvBnAct(in_channels[1], in_channels[1], kernel_size=3, stride=2, activation=activation)
        self.CSP_n4 = CSPLayer(in_channels[1], in_channels[2], n=3, depthwise=depthwise, activation=activation)


    def forward(self, x):
        inputs = [x[input] for input in self.inputs]
        c3, c4, c5 = inputs
        print(c3.shape)

        p5 = self.lateral_conv(c5) # 1024 -> 512 /2**5
        m4 = self.upsample(p5) # 512 /2**4
        m4 = torch.cat((m4, c4), dim=1) # 512 -> 1024 /2**4
        m4 = self.CSP_p4(m4) # 1024 -> 512 /2**4

        p4 = self.topdown_conv(m4) # 512 -> 256 /2**4
        m3 = self.upsample(p4) # 256 /2**3
        m3 = torch.cat((m3, c3), dim=1) # 256 -> 512 /2**3
        out1 = self.CSP_p3(m3) # 512 -> 256 /2**3

        n4 = self.bottomup_conv1(out1) # 256 -> 256 /2**4
        n4 = torch.cat((n4, p4), dim=1) # 256 -> 512 /2**4
        out2 = self.CSP_n3(n4) # 512 -> 256 / 2**4

        n5 = self.bottomup_conv2(out2) # 256 -> 256 / 2**5
        n5 = torch.cat((n5, p5), dim=1) # 256 -> 512 / 2**5
        out3 = self.CSP_n4(n5) # 512 -> 1024 / 2**5

        return (out1, out2, out3)


class YOLOv4Head(nn.Module):
    """YOLOv4 Anchor based Head
    """
    def __init__(self, 
                in_channels: tuple = (256, 512, 1024),
                num_classes: int = 80,
                anchors: List = [[[12, 16], [19, 36], [40, 28]],
                                [[36, 75], [76, 55], [72, 146]],
                                [[142, 110], [192, 243], [459, 401]]]):
        super().__init__()
        self.in_channels = in_channels
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_outputs = 5 + num_classes
        self.num_levels = len(in_channels)

    @staticmethod
    def _decode_bbox_to_xywh(bbox_pred, anchor):
        raise NotImplementedError

    def forward(self, x):
        assert len(x) == self.num_levels

    def forward_single(self, x: torch.Tensor, conv: nn.Module):
        """Forward fetaure of single scale"""
        out = conv(x)
        bs, _, h, w = out.shape
        out = out.view(bs, self.num_anchors, h, w, -1)
        class_score = out[..., 5:]
        bbox_pred = out[..., :4]
        confidence_score = out[...,4:5]
        return class_score, bbox_pred, confidence_score

class YOLOP(nn.Module):
    """YOLO for Panoptic Driving Perception
        Note: 
            - The original YOLOP utilizes FPN but we will be testing between FPN and PANet to see which provides better accuracy 
            and see the tradeoffs between accuracy and speed
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError   