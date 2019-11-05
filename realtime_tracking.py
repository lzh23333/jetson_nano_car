# realtime tracking, detect the first human and tracking
import cv2
import numpy as np
import torch
from os.path import join
from os import listdir

from DaSiamRPN.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.net import SiamRPNvot
from DaSiamRPN.utils import cxy_wh_2_rect, rect_2_cxy_wh
from Yolov3.detector import detector


# init yolov3, video
yolo = detector()
detect_human = False
cap = cv2.VideoCapture(0)

# init tracking 
net = SiamRPNvot()
net.load_state_dict(torch.load('./DaSiamRPN/SiamRPNVOT.model'))
net.eval().cuda()

init_box = []

while not detect_human:
    _, img = cap.read()
    human = yolo.detect_human(img)
    print(human)
    if len(human) == 0:
        continue
    else:
        detect_human = True
        init_box = [human[0], human[1], human[2]-human[0], human[3]-human[1]]
print('human detected ! start initialize SiamRPN')

target_pos, target_sz = rect_2_cxy_wh(init_box)
state = SiamRPN_init(img, target_pos, target_sz, net)

while True:
    _, img = cap.read()
    state = SiamRPN_track(state, img)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(img, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', img)
    cv2.waitKey(1)



    

