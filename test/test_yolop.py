import torch
import unittest

from lumen.models.yolop import PANet, YOLOv4Head, SegmentationHead, YOLOP
from lumen.models.darknet import CSPDarknet53

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestYOLOP(unittest.TestCase):
    def testPANet(self):
        x = torch.randn(1, 3, 416, 416).to(device)
        darknet = CSPDarknet53(in_channels=3).to(device)
        #darknet = torch.compile(darknet)
        panet = PANet().to(device)
        #panet = torch.compile(panet)
        out = darknet(x)
        out = panet(out)

        assert out[0].shape == (1, 256, 52, 52)
        assert out[1].shape == (1, 512, 26, 26)
        assert out[2].shape == (1, 1024, 13, 13)

    def testYOLOHead(self):
        x = torch.randn(1, 3, 416, 416).to(device)
        darknet = CSPDarknet53(in_channels=3).to(device)
        panet = PANet().to(device)
        head = YOLOv4Head(num_classes=10).to(device)
        out = darknet(x)
        out = panet(out)
        class_pred, bbox_pred, confidence_pred = head(out)

        assert class_pred[0].shape == (1, 3, 52, 52, 10)
        assert class_pred[1].shape == (1, 3, 26, 26, 10)
        assert class_pred[2].shape == (1, 3, 13, 13, 10)
        assert bbox_pred[0].shape == (1, 3, 52, 52, 4)
        assert bbox_pred[1].shape == (1, 3, 26, 26, 4)
        assert bbox_pred[2].shape == (1, 3, 13, 13, 4)
        assert confidence_pred[0].shape == (1, 3, 52, 52, 1)
        assert confidence_pred[1].shape == (1, 3, 26, 26, 1)
        assert confidence_pred[2].shape == (1, 3, 13, 13, 1)

    def testSegmentationHead(self):
        x = torch.randn(1, 256, 52, 52).to(device)
        model = SegmentationHead(in_channels=256 ,num_classes=10).to(device)
        out = model(x)
        assert out.shape == (1, 10, 416, 416)

    def testYOLOP(self):
        x = torch.randn(1, 3, 416, 416).to(device)
        model = YOLOP(in_channels=3, num_det_classes=10, num_seg_classes=10).to(device)
        detection, segmentation = model(x)
        class_pred, bbox_pred, confidence_pred = detection
        assert class_pred[0].shape == (1, 3, 52, 52, 10)
        assert class_pred[1].shape == (1, 3, 26, 26, 10)
        assert class_pred[2].shape == (1, 3, 13, 13, 10)
        assert bbox_pred[0].shape == (1, 3, 52, 52, 4)
        assert bbox_pred[1].shape == (1, 3, 26, 26, 4)
        assert bbox_pred[2].shape == (1, 3, 13, 13, 4)
        assert confidence_pred[0].shape == (1, 3, 52, 52, 1)
        assert confidence_pred[1].shape == (1, 3, 26, 26, 1)
        assert confidence_pred[2].shape == (1, 3, 13, 13, 1) 
        assert segmentation.shape == (1, 10, 416, 416)

if __name__ == '__main__':
    unittest.main()