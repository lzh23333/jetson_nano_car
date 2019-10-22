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
    # load model 
    net = SiamRPNvot()
    net.load_state_dict(torch.load('./DaSiamRPN/SiamRPNVOT.model'))
    net.eval().cuda()
    

    # image and init box
    dataset_folder = './test_video/Panda'
    ground_truth = np.loadtxt(
        join(dataset_folder, 'groundtruth_rect.txt'),
        delimiter=','
    )
    init_box = ground_truth[0, :]
    image_list = [join(dataset_folder, 'img', x) for x in list(listdir(join(dataset_folder, 'img')))]

    # tracker init
    target_pos, target_sz = rect_2_cxy_wh(init_box)
    im = cv2.imread(image_list[0])
    state = SiamRPN_init(im, target_pos, target_sz, net)

    toc = 0
    # tracking and visualization
    for image_name in image_list[1:]:
        im = cv2.imread(image_name)
        tic = cv2.getTickCount()
        state = SiamRPN_track(state, im)  # track
        toc += cv2.getTickCount()-tic
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        cv2.imshow('SiamRPN', im)
        cv2.waitKey(1)

    print('Tracking Speed {:.1f}fps'.format((len(image_list)-1)/(toc/cv2.getTickFrequency())))




if __name__ == '__main__':
    main()
