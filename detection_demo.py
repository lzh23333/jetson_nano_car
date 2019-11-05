import cv2
from Yolov3.detector import detector
import torch


if __name__ == "__main__":
    yolo = detector()
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        detection = yolo.detect(img.copy())
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            name = yolo.coco_names[str(int(cls_pred))]
            print(name)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
            cv2.putText(img, name, (x1, y1), font, 0.5, (0, 255, 0))
        
        cv2.imshow('test', img)
        cv2.waitKey(1)