# camera demo
# a simple demo using DaSiamRPN model
import cv2
import numpy as np
import torch
from os.path import join
from os import listdir

from DaSiamRPN.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.net import SiamRPNvot
from DaSiamRPN.utils import cxy_wh_2_rect, rect_2_cxy_wh

def main():
    cap = cv2.VideoCapture(0)
    net = SiamRPNvot()
    net.load_state_dict(torch.load('./DaSiamRPN/SiamRPNVOT.model'))
    net.eval().cuda()
    
    _, im = cap.read()
    init_box = cv2.selectROI('selectROI', im)
    print(init_box)
    target_pos, target_sz = rect_2_cxy_wh(init_box)
    state = SiamRPN_init(im, target_pos, target_sz, net)

    while True:
        _, im = cap.read()
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        cv2.imshow('SiamRPN', im)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()  