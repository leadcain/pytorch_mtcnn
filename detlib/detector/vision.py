#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def easy_vis(img, bboxes, lmks, scale=(1,1),  threshold=0.9, save_path=None):
    #- Nothing to display
    if len(bboxes) == 0:
        return img

    #- remove bboxes cls_prob. <  threshold
    conf_idx = np.asarray(np.where(bboxes[:,4] > threshold)).reshape(-1)
    bboxes = bboxes[conf_idx]

    #- x,y sclale
    bboxes[:,0] = bboxes[:,0] * scale[0]
    bboxes[:,1] = bboxes[:,1] * scale[1]
    bboxes[:,2] = bboxes[:,2] * scale[0]
    bboxes[:,3] = bboxes[:,3] * scale[1]

    #- visualize all faces
    for box in bboxes:
        cv2.rectangle(img, 
                     (box[0].astype(int), box[1].astype(int)),
                     (box[2].astype(int), box[3].astype(int)), 
                     (0, 255, 255), 2)

        cv2.putText(img, '{:.2f}'.format(box[4]),
                    org=(int(box[0]), int(box[1]-10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=2)

    #- visualize all landmarks
    if lmks is not None:
        lmks = lmks.reshape(-1, 6, 2)
        lmks = lmks[conf_idx]

        lmks[:,:,0] = lmks[:,:,0] * scale[0]
        lmks[:,:,1] = lmks[:,:,1] * scale[1]
        lmks = lmks.astype('int')

        for lmk in lmks:
            cv2.circle(img, (lmk[0,0], lmk[0,1]), radius=3, thickness=2, color=(0, 0, 255))
            cv2.circle(img, (lmk[1,0], lmk[1,1]), radius=3, thickness=2, color=(0, 0, 255))
            cv2.circle(img, (lmk[2,0], lmk[2,1]), radius=3, thickness=2, color=(0, 0, 255))
            cv2.circle(img, (lmk[3,0], lmk[3,1]), radius=3, thickness=2, color=(0, 0, 255))
            cv2.circle(img, (lmk[4,0], lmk[4,1]), radius=3, thickness=2, color=(0, 0, 255))
            cv2.circle(img, (lmk[5,0], lmk[5,1]), radius=3, thickness=2, color=(0, 0, 255))

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return img


