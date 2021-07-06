# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import random
import argparse
import numpy as np
sys.path.append(os.getcwd())
import detlib.detector.utils as utils
from detlib.detector.landmark_utils import *

DBG = False

def gen_data(args, argument):

    image_id = 0
    if args.img_size == 12:
        folder_name = 'pnet'
    elif args.img_size == 24:
        folder_name = 'rnet'
    elif args.img_size == 48:
        folder_name = 'onet'
    else:
        raise TypeError('img_size must be 12, 24 or 48')
    print(folder_name)

    txt_save_dir = os.path.join(args.save_dir, 'anno_store/%s' % folder_name)
    img_save_dir = os.path.join(args.save_dir, '%s/landmark' % folder_name)
    for folder in [txt_save_dir, img_save_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    save_txt_name = 'landmark_%d.txt' % args.img_size
    save_txt_to   = os.path.join(txt_save_dir, save_txt_name)

    with open(args.anno_file, 'r') as f2:
        annotations = f2.readlines()
    f2.close()

    num = len(annotations)
    print("%d total images" % num)

    l_idx = 0
    f = open(save_txt_to, 'w')

    for idx, annotation in enumerate(annotations):
        f_img_list = []
        f_lmk_list = []

        annotation = '/'.join(annotation.strip().split('\\')).split(' ')

        im_path     = os.path.join(args.data_dir, annotation[0])
        gt_box      = np.array(list(map(float, annotation[1:5])), dtype=np.float32)
        
        img = cv2.imread(im_path)

        assert (img is not None)

        height, width, channel = img.shape

        if (idx + 1) % 100 == 0:
            print("%d images done, landmark images: %d" % (idx+1, l_idx))

        x1, x2, y1, y2 = gt_box        # raw bbox[x1, x2, y1, y2]
        gt_box[1], gt_box[2] = y1, x2  # re-order[x1, y1, x2, y2]
        
        gt_w = x2 - x1 + 1
        gt_h = y2 - y1 + 1
 
        if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
            #skip smaller face
            continue

        cropped_im = img[int(y1):int(y2)+1, int(x1):int(x2)+1, :]
        resized_im = cv2.resize(cropped_im, (args.img_size, args.img_size),interpolation=cv2.INTER_LINEAR)

        """ 
            landmark label info:
            12-landmark point [left-eye-x, left-eye-y, right-eye-x, right-eye-y, 
                               nose-x, nose-y,
                               left-mou-x, left-mou-y, right-mou-x, right-mou-y,
                               chin-x, chin-y]

            (6,2) reshaped landmark
                [[ left-eye-x,  left-eye-y],
                 [right-eye-x, right-eye-y],
                 [     nose-x,      nose-y],
                 [ left-mou-x,  left-mou-y],
                 [right-mou-x, right-mou-y],
                 [     chin-x,      chin-y]]
        """
        
        #- raw image, label append without augumentation
        f_img_list.append(resized_im)

        bbox = BBox([x1,y1,x2,y2])

        if argument:
            #####################################
            #- Aug1 : ramdom shift  
            #####################################
            for i in range(args.num_rands):
                bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.85), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x   = np.random.randint(-gt_w * 0.1, gt_w * 0.1)
                delta_y   = np.random.randint(-gt_h * 0.1, gt_h * 0.1)

                nx1 = max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size

                if nx2 > width or ny2 > height:
                    continue

                nx1 = int(nx1)
                nx2 = int(nx2)
                ny1 = int(ny1)
                ny2 = int(ny2)

                crop_box   = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(bbox_size)
                offset_y1 = (y1 - ny1) / float(bbox_size)
                offset_x2 = (x2 - nx2) / float(bbox_size)
                offset_y2 = (y2 - ny2) / float(bbox_size)


                cropped_im = img[ny1:ny2+1, nx1:nx2+1, :]
                resized_im = cv2.resize(cropped_im, (args.img_size, args.img_size),interpolation=cv2.INTER_LINEAR)

                iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
                if iou > args.threshold:
                    f_img_list.append(resized_im)

                    bbox = BBox([nx1,ny1,nx2,ny2])
                    #####################################
                    #- Aug2 : Flip  
                    #####################################
                    if random.choice([0,1]) > 0:
                        flipped_im = flip(resized_im)
                        f_img_list.append(flipped_im)
 
                    #####################################
                    #- Aug3 : Rotate  
                    #####################################
                    if random.choice([0,1]) > 0:
                        rotated_im = rotate(img, bbox, random.randint(5, 45)) 
                        rotated_im  = cv2.resize(rotated_im, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                        f_img_list.append(rotated_im)
                      
                        #- flip of rotated
                        rotated_flipped_im = flip(rotated_im) 
                        
                        #- save image
                        f_img_list.append(rotated_flipped_im)

                    if random.choice([0,1]) > 0:
                        inv_rotated_im = rotate(img, bbox, -random.randint(5, 45))
                        inv_rotated_im = cv2.resize(inv_rotated_im, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                        
                        #- save image, landmark
                        f_img_list.append(inv_rotated_im)

                        #- flip of inversed rotated
                        inv_rotated_flipped_im = flip(inv_rotated_im)
                       
                        #- save image, landmark 
                        f_img_list.append(inv_rotated_flipped_im)

            f_imgs = np.asarray(f_img_list)

            #- added by bjkim
            for i in range(len(f_imgs)):
                save_file = os.path.join(img_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, f_imgs[i])

                #f.write(save_file + ' -2 ' + ' '.join(gt_box)+'\n')
                f.write(save_file + ' -2 ' + ' ' + str(offset_x1) + ' ' + str(offset_y1) + ' ' + str(offset_x2) + ' ' + str(offset_y2) +'\n')
                l_idx += 1

    f.close()


def gen_config():
    parser = argparse.ArgumentParser(description=' Generate lmk file')
    parser.add_argument('--root_dir',  type=str,  default='/datasets/WIDER_FACE')
    parser.add_argument('--anno_file', type=str,  default='/datasets/WIDER_FACE/face_landmark/TrainImageList_6Landmark.txt')
    parser.add_argument('--data_dir',  type=str,  default='/datasets/WIDER_FACE/face_landmark')
    parser.add_argument('--save_dir',  type=str,  default='/datasets/WIDER_FACE')
    parser.add_argument('--img_size',  type=int,  default=24)    
    parser.add_argument('--num_rands', type=int,  default=30)    
    parser.add_argument('--threshold', type=float,default=0.7)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    gen_data(gen_config(), argument=True)
