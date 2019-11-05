from Yolov3.model import Darknet
import torch
import cv2
from torchvision import transforms
from Yolov3.utils import pad_to_square, resize
from Yolov3.utils import non_max_suppression, rescale_boxes
from torch.autograd import Variable


class detector(object):

    def __init__(self):
        cfg_file = './Yolov3/yolov3.cfg'
        weights = './Yolov3/models/yolov3.weights'
        coco_name = './Yolov3/coco.names'
        self.img_size = 416

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(cfg_file).to(device)
        self.model.load_darknet_weights(weights)
        self.model.eval()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        with open(coco_name, 'r') as f:
            self.coco_names = {}
            for label, line in enumerate(f):
                self.coco_names[str(label)] = line
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), ]
        )


    def detect(self, img):
        # img is numpy array
        # reshape and toTensor
        origin_size = img.shape[:2]
        img = self.transforms(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size).unsqueeze(0)
        img = Variable(img.type(self.Tensor))
        print(img.shape)
        detections = self.model(img)
        detections = non_max_suppression(detections)[0]
        detections = rescale_boxes(detections, 416, origin_size)
        return detections
