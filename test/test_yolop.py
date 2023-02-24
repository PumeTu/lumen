import torch
import unittest

from lumen.models.yolop import PANet
from lumen.models.darknet import CSPDarknet53

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestPANet(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()