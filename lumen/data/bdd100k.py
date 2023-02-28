import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class BDD100k(data.Dataset):
    def __init__(self, root: str = '../bdd100k/', transforms: transforms = None, training: bool = False, input_size: tuple = (416, 416), anchors: List = [[[12, 16], [19, 36], [40, 28]],
                                                                                                                        [[36, 75], [76, 55], [72, 146]],
                                                                                                                        [[142, 110], [192, 243], [459, 401]]]):
        super().__init__()
        self.root = root
        self.training = training
        self.transforms = transforms
        self.anchors = torch.tensor(anchors)
        self.scale = len(anchors)
        self.input_size = input_size
        self.image_path = self.root + 'images/100k/train/' if self.training else self.root + 'images/100k/val/'
        self.det_path = self.root + 'labels/det_20/det_train.json' if self.training else self.root + 'labels/det_20/det_val.json'
        self.lane_path = self.root + 'labels/lane/masks/train/' if self.training else self.root + 'labels/lane/masks/val/'
        self.drivable_path = self.root + 'labels/drivable/masks/train/' if self.training else self.root + 'labels/drivable/masks/val/'
        
        detections = pd.read_json(self.det_path)
        attributes = pd.DataFrame.from_records(detections.attributes)
        self.detections = pd.concat([detections.drop(labels='attributes', axis=1), attributes], axis=1)
        self.detections.dropna(axis=0, inplace=True)

        self.classes = {
            "pedestrian":1,
            "rider":2,
            "car":3,
            "truck":4,
            "bus":5,
            "train":6,
            "motorcycle":7,
            "bicycle":8,
            "traffic light":9,
            "traffic sign": 10,
            "other vehicle": 11,
            "other person": 12,
            "trailer": 13
        }
        
        self.num_outputs = len(self.classes) + 5
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index):
        target = self.detections.iloc[index]
        image = torchvision.io.read_image(self.image_path + target['name'])
        annotations = target['labels']
        labels = []
        for object in annotations:
            label = list(object['box2d'].values())
            label.append(self.classes[object['category']])
            labels.append(label)

        lane_path = self.lane_path + target['name'].replace('.jpg', '.png')
        drivable_path = self.drivable_path + target['name'].replace('.jpg', '.png')

        target = {}
        target['detections'] = torch.as_tensor(labels, dtype=torch.float32)
        target['segmentations'] = self._build_seg_target(lane_path, drivable_path)

        if self.transforms:
            image = self.transforms(image)

        return image, target
       
    def _build_seg_target(self, lane_path, drivable_path):
        """Build groundtruth for  segmentation
        Note: This combines the lanes and drivable masks into one 
        Args:
            lane_path (str): path to lane binary mask
            drivable_path (str): path to drivable binary mask
        """
        lane = cv2.imread(lane_path)[..., 0]
        drivable = cv2.imread(drivable_path)[..., 0]

        lanes = np.bitwise_and(lane, 0b111)
        lane_mask, drivable_mask = [], []
        for i in range(9):
            lane_mask.append(np.where(lanes==i, 1, 0))
            if i in range(3):
                drivable_mask.append(np.where(drivable==i, 1, 0))
        lane_mask, drivable_mask = np.stack(lane_mask), np.stack(drivable_mask)
        mask = np.concatenate((lane_mask, drivable_mask), axis=0)
        
        return torch.tensor(mask)
    
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), targets