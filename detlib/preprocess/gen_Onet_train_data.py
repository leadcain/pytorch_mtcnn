# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import argparse
import numpy as np
import shutil
import argparse

sys.path.append(os.getcwd())
from six.moves import cPickle
from detlib.detector.imagedb import ImageDB
from detlib.detector.utils import convert_to_square,IoU
from detlib.detector.image_reader import TestImageLoader
from detlib.detector.detect import DetectFace 


def gen_onet_data(args):

    mtcnn_detector = DetectFace(pnet=args.pnet_weight, 
                                rnet=args.rnet_weight,
                                min_face=12, device=args.device)

    anno_dir = os.path.join(args.root_dir, args.anno_dir)
    img_dir  = os.path.join(args.root_dir, args.img_dir)
    anno_file = os.path.join(anno_dir, 'wider_face_train.txt')
    imagedb = ImageDB(anno_file, args.root_dir, mode='test', prefix_path='')
    imdb    = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)
    print('size:%d' % image_reader.size) # still use wideface dataset

    all_boxes, batch_idx = list(), 0
    for databatch in image_reader:

        if (batch_idx + 1) % 100 == 0:
            print("%d images done" % (batch_idx + 1))

        im = databatch

        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        if p_boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue

        # rnet detection
        r_boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if r_boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue

        all_boxes.append(r_boxes_align)
        batch_idx += 1

    save_path = os.path.join(anno_dir, 'onet')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print("Remove onet anno_store old files")
        shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    
    gen_onet_sample_data(args, args.root_dir, img_dir, anno_dir, save_file)


def gen_onet_sample_data(args, data_dir, img_dir, anno_dir, det_boxs_file):

    neg_save_dir  = os.path.join(data_dir, "train_data/train_onet/negative")
    pos_save_dir  = os.path.join(data_dir, "train_data/train_onet/positive")
    part_save_dir = os.path.join(data_dir, "train_data/train_onet/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            print("Remove train_onet old images")
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    anno_file = os.path.join(anno_dir, 'wider_face_train.txt')
    
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:

        annotation = annotation.strip().split(' ')
        im_idx = os.path.join('',annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = os.path.join(anno_dir, 'onet')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue

        #added by bjkim
        img_path = os.path.join(img_dir, im_idx+'.jpg')
        img = cv2.imread(img_path)

        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 40 or height < 40 \
                 or x_left < 0 or y_top < 0 or \
                 (x_right > img.shape[1] - 1) or (y_bottom > img.shape[0] - 1):
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < args.iou_neg:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left)   / float(width)
                offset_y1 = (y1 - y_top)    / float(height)
                offset_x2 = (x2 - x_right)  / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= args.iou_pos:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.4f %.4f %.4f %.4f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= args.iou_part:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.4f %.4f %.4f %.4f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Gen Onet Data')
    parser.add_argument('--root_dir',    type=str,   default='/datasets/WIDER', help='Dataset Root Path')
    parser.add_argument('--img_dir',     type=str,   default='WIDER_train/images', help='Rnet data path')
    parser.add_argument('--anno_dir',    type=str,   default='anno_store', help='Annotation path')
    parser.add_argument('--pnet_weight', type=str,   default='./model/checkout/pnet/pnet_epoch_20.pt')
    parser.add_argument('--rnet_weight', type=str,   default='./model/checkout/rnet/rnet_epoch_20.pt')
    parser.add_argument('--device',      type=str,   default='cuda', help='cuda or cpu')
    parser.add_argument('--iou_pos',     type=float, default=0.65, help='pos iou threshold')
    parser.add_argument('--iou_neg',     type=float, default=0.25, help='neg iou threshold')
    parser.add_argument('--iou_part',    type=float, default=0.4, help='part iou threshold')

    args = parser.parse_args()
    return args


def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"


if __name__ == '__main__':

    gen_onet_data(parse_args())




