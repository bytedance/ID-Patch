# Copyright (c) OpenMMLab
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on October 10, 2024.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/open-mmlab/mmpose/blob/main/LICENSE.
#
# This modified file is released under the same license.


import numpy as np
import cv2
from itertools import product
import math

def draw_openpose_from_mmpose(canvas, keypoints, keypoint_scores, kpt_thr=0.3, ignore_individual_points=False):
    """
        canvas: background image
        keypoints: N x 17 x 2
        keypoint_scores: N x 17
        ret: RGB order (note: although we use cv2 to process image, result is in RGB order)
    """
    
    # openpose format
    limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85]]

    stickwidth = 4
    num_openpose_kpt = 18
    num_link = len(limb_seq)
    
    # concatenate scores and keypoints
    keypoints = np.concatenate((keypoints, keypoint_scores.reshape(-1, 17, 1)), axis=-1)
    
    # compute neck joint
    neck = (keypoints[:, 5] + keypoints[:, 6]) / 2
    #if keypoints[:, 5, 2] < kpt_thr or keypoints[:, 6, 2] < kpt_thr:
    #    neck[:, 2] = 0
    neck[:, 2] = np.minimum(keypoints[:, 5, 2], keypoints[:, 6, 2])

    # 17 keypoints to 18 keypoints
    new_keypoints = np.insert(keypoints[:, ], 17, neck, axis=1)

    # mmpose format to openpose format
    openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
    mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints[:, openpose_idx, :] = new_keypoints[:, mmpose_idx, :]
    
    black_img = canvas
    num_instance = new_keypoints.shape[0]
    
    # drw keypoints
    for i in range(num_instance):
        valid = [False] * 18
        for link_idx in range(num_link):
            conf = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 2]
            if np.sum(conf > kpt_thr) == 2:
                valid[limb_seq[link_idx][0]-1] = True
                valid[limb_seq[link_idx][1]-1] = True
        for j in range(num_openpose_kpt):
            x, y, conf = new_keypoints[i][j]
            if conf > kpt_thr and valid[j]:
                cv2.circle(black_img, (int(x), int(y)), 4, colors[j], thickness=-1)

    # draw links
    cur_black_img = black_img.copy()
    for i, link_idx in product(range(num_instance), range(num_link)):
        conf = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 2]
        if np.sum(conf > kpt_thr) == 2:
            Y = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 0]
            X = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle),
                0, 360, 1)
            cv2.fillConvexPoly(cur_black_img, polygon, colors[link_idx])
    black_img = cv2.addWeighted(black_img, 0.4, cur_black_img, 0.6, 0)
    
    return black_img