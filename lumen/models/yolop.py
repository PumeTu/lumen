import torch
import torch.nn as nn

class PaNet(nn.Module):
    """Path Aggregation Network
    Args:

    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_bottom_up():
        raise NotImplementedError
    
    @staticmethod
    def _build_top_down():
        raise NotImplementedError

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