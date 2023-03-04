import torch
from torch.utils.data import DataLoader
import unittest

from lumen.data.bdd100k import BDD100k

class TestBDD100k(unittest.TestCase):
    def test_single_process_dataloader(self):
        train_dataset = BDD100k(root='/home/pumetu/data/bdd100k/', training=True)
        self._check_dataloader(train_dataset, num_workers=0)     
        test_dataset = BDD100k(root='/home/pumetu/data/bdd100k/', training=False)
        self._check_dataloader(test_dataset, num_workers=0)

    def test_multi_process_dataloader(self):
        train_dataset = BDD100k(root='/home/pumetu/data/bdd100k/', training=True)
        self._check_dataloader(train_dataset, num_workers=2)
        test_dataset = BDD100k(root='/home/pumetu/data/bdd100k/', training=False)
        self._check_dataloader(test_dataset, num_workers=2)

    def _check_dataloader(self, data, num_workers):
        """This function only tests that the loading process throws no error"""
        loader = DataLoader(data, batch_size=4, num_workers=num_workers)
        for _ in loader:
            pass

    # def test_target(self):
    #     dataset = BDD100k(root='/home/pumetu/data/bdd100k/', training=False)
    #     loader = DataLoader(dataset, batch_size=4)
    #     for img, target in loader:


