import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from typing import List
import cv2
import numpy as np

from utils.bbox import xyxy_to_xywh, norm_bbox


class BDD100k(data.Dataset):
    def __init__(self, root: str = '../bdd100k/', transforms: transforms = None, training: bool = False, input_size: tuple = (416, 416), anchors: List = [[[12, 16], [19, 36], [40, 28]],
                                                                                                                        [[36, 75], [76, 55], [72, 146]],
                                                                                                                        [[142, 110], [192, 243], [459, 401]]]):
        self.root = root
        self.training = training
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
 
        labels = torch.Tensor(labels)
        detections = self._build_yolo_targets(labels)
        segmentations = self._build_seg_target(lane_path, drivable_path)

        return image, detections, segmentations

    @staticmethod
    def _iou(box, anchor):
        intersection = torch.min(box[...,0], anchor[..., 0]) * torch.min(box[...,1], anchor[...,1])
        union = box[..., 0] * box[..., 1] + anchor[..., 0] * anchor[...,1] - intersection
        return intersection / union
    
    def _build_yolo_targets(self, labels):
        """
        Builds the groundtruth for a YOLO network given the bounding box and class labels
        Args:
            labels (Tensor): bounding box and class labels of the entire dataset
                - shape: (N, 5)
        Returns:
            (det1, det2, det3): all 3 detection targets for the 3 scales
                - shape: (n_anchors, h / 2**n, w / 2**n, n_output)
                n_output: conf, x, y, w, h, classes
        """
        groundtruths = [torch.zeros((self.scale, int(self.input_size[0] / 2**x), int(self.input_size[1] / 2**x), self.num_outputs)) for x in reversed(range(3, 6))]
        for label in labels:
            iou_index = self._iou(label[:5], self.anchors).argmax()
            scale = iou_index // 3
            anchor_index = iou_index % 3
            _, sy, sx, _ = groundtruths[scale].shape
            bbox = norm_bbox(xyxy_to_xywh(label[:5]), *self.input_size)
            i, j = int(sy * bbox[1]), int(sx * bbox[0])
            if not groundtruths[scale][anchor_index, i, j, self.num_classes]:
                groundtruths[scale][anchor_index, i, j, self.num_classes] = 1
                y_cell, x_cell = sy * bbox[1] - i, sx * bbox[0] - j
                w_cell, h_cell = bbox[2] * sx, bbox[3] * sy
                box = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                groundtruths[scale][anchor_index, i, j, self.num_classes+1:self.num_classes+5] = box
                obj_class = int(label[-1].item())
                groundtruths[scale][anchor_index, i, j, obj_class] = 1

        return groundtruths
                
    def _build_seg_target(self, lane_path, drivable_path):
        """
        Build groundtruth for  segmentation
            - this will combine the lanes and drivable masks into one 
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