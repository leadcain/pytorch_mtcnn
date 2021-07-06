# -*- coding: utf-8 -*-

import os
import cv2
import sys
import numpy as np
import shutil
import argparse
import random
sys.path.append(os.getcwd())
from detlib.detector.utils import IoU
from detlib.detector.landmark_utils import rotate, BBox


def gen_pnet_data(args):
    img_dir = os.path.join(args.root_dir, args.img_dir)
    part_save_dir = os.path.join(args.root_dir, args.part_save_dir)
    pos_save_dir  = os.path.join(args.root_dir, args.pos_save_dir)
    neg_save_dir  = os.path.join(args.root_dir, args.neg_save_dir)
    anno_dir = os.path.join(args.root_dir, args.anno_dir)


    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            print("Remove old images")
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    
    # store labels of positive, negative, part images
    if not os.path.exists(os.path.join(anno_dir, 'pnet')):
        os.makedirs(os.path.join(anno_dir, 'pnet'))
    
    anno_file = os.path.join(anno_dir, 'wider_face_train.txt')
    f_pos = open(os.path.join(anno_dir, 'pnet/pos_12.txt'), 'w')
    f_neg = open(os.path.join(anno_dir, 'pnet/neg_12.txt'), 'w')
    f_part = open(os.path.join(anno_dir, 'pnet/part_12.txt'), 'w')
    
    # anno_file: store labels of the wider face training data
    with open(anno_file, 'r') as f:
         annotations = f.readlines()
    
    num = len(annotations)
    print("%d pics in total" % num)   # 12880
    
    default_num_pos_rotate = 30
    print_freq      = 100
    
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # positive, negative, dont care
    idx   = 0
    
    for anno_idx, annotation in enumerate(annotations):
        annotation = annotation.strip().split(' ')
        im_path = os.path.join(img_dir, annotation[0])
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    
        print(im_path)
        img = cv2.imread(im_path + '.jpg')
        idx += 1
    
        if (idx + 1) % print_freq == 0:
            print(idx + 1, "images done")
        
        height, width, channel = img.shape
    
        neg_num = 0
        while neg_num < args.num_neg:
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
    
            Iou = IoU(crop_box, boxes)
    
            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < args.iou_neg:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                save_flag = cv2.imwrite(save_file, resized_im)
                if save_flag:
                    f_neg.write(save_file + ' 0\n')
                    n_idx += 1
                    neg_num += 1
    
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
    
            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            #if min(w, h)<= 0 or max(w, h) < 40 or x1 < 0 or y1 < 0:
            if w < 40 or h < 40 or x1 < 0 or y1 < 0:
                continue
    
            # generate negative examples that have overlap with gt
            for i in range(args.num_neg):
                size = np.random.randint(12, min(width, height) / 2)
    
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)
    
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
    
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                #- neg
                if np.max(Iou) < args.iou_neg:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    save_flag = cv2.imwrite(save_file, resized_im)
                    if save_flag:
                        f_neg.write(save_file + ' 0\n')
                        n_idx += 1
    
            # generate positive examples and part faces
            for i in range(args.num_pos):
                size = np.random.randint(int(max(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
    
                # delta here is the offset of box center
                try:
                    delta_x = np.random.randint(-w * 0.2, w * 0.2)
                    delta_y = np.random.randint(-h * 0.2, h * 0.2)
                except Exception as e:
                    print(e)
                    break
    
                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size
    
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
    
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
    
                cropped_im = img[int(ny1):int(ny2), int(nx1):int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
                box_ = box.reshape(1, -1)
                #- pos
                if IoU(crop_box, box_) >= args.iou_pos:
                    save_file = os.path.join(pos_save_dir, "%s_%s.jpg" % (im_path.split('/')[-1], p_idx))
                    save_flag = cv2.imwrite(save_file, resized_im)
                    if save_flag:
                        f_pos.write(save_file + ' 1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        p_idx += 1
                #- part
                #elif IoU(crop_box, box_) >= 0.3 and IoU(crop_box, box_) <= 0.5:
                elif IoU(crop_box, box_) >= args.iou_part:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    save_flag = cv2.imwrite(save_file, resized_im)
                    if save_flag:
                        f_part.write(save_file + ' -1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        d_idx += 1
    
            for i in range(args.num_rot):
                size = np.random.randint(int(max(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
    
                try:
                    delta_x = np.random.randint(-w * 0.2, w * 0.2)
                    delta_y = np.random.randint(-h * 0.2, h * 0.2)
                except Exception as e:
                    print(e)
                    break
    
                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size
    
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
    
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
    
                bbox = BBox([int(nx1), int(ny1), int(nx2), int(ny2)])
    
                #-  clock-wise
                rotate_im = rotate(img, bbox, random.randint(1, 10))
                rotate_im = cv2.resize(rotate_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
                box_ = box.reshape(1, -1)
                #- pos
                if IoU(crop_box, box_) >= args.iou_pos:
                    save_file = os.path.join(pos_save_dir, "%s_%s.jpg" % (im_path.split('/')[-1], p_idx))
                    save_flag = cv2.imwrite(save_file, rotate_im)
                    if save_flag:
                        f_pos.write(save_file + ' 1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        p_idx += 1
    
                elif IoU(crop_box, box_) >= args.iou_part:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    save_flag = cv2.imwrite(save_file, rotate_im)
                    if save_flag:
                        f_part.write(save_file + ' -1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        d_idx += 1
    
                #- inversed clock-wise
                rotate_im = rotate(img, bbox, -random.randint(1, 10))
                rotate_im = cv2.resize(rotate_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
                if IoU(crop_box, box_) >= args.iou_pos:
                    save_file = os.path.join(pos_save_dir, "%s_%s.jpg" % (im_path.split('/')[-1], p_idx))
                    save_flag = cv2.imwrite(save_file, rotate_im)
                    if save_flag:
                        f_pos.write(save_file + ' 1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        p_idx += 1
    
                #elif IoU(crop_box, box_) >= 0.3 and IoU(crop_box, box_) <= 0.5:
                elif IoU(crop_box, box_) >= args.iou_part:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    save_flag = cv2.imwrite(save_file, rotate_im)
                    if save_flag:
                        f_part.write(save_file + ' -1 %.4f %.4f %.4f %.4f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        d_idx += 1
    
    
        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
    
    f_pos.close()
    f_neg.close()
    f_part.close()
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Gen Pnet Data")
    parser.add_argument('--root_dir',      type=str,    default='/datasets/WIDER', help='Dataset root')
    parser.add_argument('--img_dir',       type=str,    default='WIDER_train/images', help='Images path of Dataset')
    parser.add_argument('--part_save_dir', type=str,    default='train_data/train_pnet/part', help='part image save path')
    parser.add_argument('--pos_save_dir',  type=str,    default='train_data/train_pnet/pos', help='pos image save path')
    parser.add_argument('--neg_save_dir',  type=str,    default='train_data/train_pnet/neg', help='neg image save path')
    parser.add_argument('--anno_dir',      type=str,    default='anno_store', help='annotation save path')
    parser.add_argument('--num_neg',       type=int,    default=60,  help='Iteration number for generating neg sample')
    parser.add_argument('--num_pos',       type=int,    default=30,  help='Iteration number for generating pos sample')
    parser.add_argument('--num_rot',       type=int,    default=10,  help='Iteration number for generating rotation sample')
    parser.add_argument('--iou_pos',       type=float,  default=0.7, help='pos sample IOU threshold')
    parser.add_argument('--iou_neg',       type=float,  default=0.3, help='neg sample IOU threshold')
    parser.add_argument('--iou_part',      type=float,  default=0.4, help='neg sample IOU threshold')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    gen_pnet_data(parse_args())











