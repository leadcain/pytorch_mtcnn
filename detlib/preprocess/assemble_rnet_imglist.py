# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import argparse
import detlib.preprocess.assemble as assemble


def gen_assemble_data(args):
    anno_dir = os.path.join(args.root_dir, args.anno_dir) 
    
    rnet_pos_file     = os.path.join(anno_dir, 'rnet/pos_24.txt')
    rnet_part_file    = os.path.join(anno_dir, 'rnet/part_24.txt')
    rnet_neg_file     = os.path.join(anno_dir, 'rnet/neg_24.txt')
    #rnet_landmark_file    = os.path.join(anno_dir, 'rnet/landmark_24.txt') 
    
    train_file  = os.path.join(anno_dir, 'rnet/train_anno_24.txt')
    eval_file   = os.path.join(anno_dir, 'rnet/eval_anno_24.txt')

    anno_list = []

    anno_list.append(rnet_pos_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    #anno_list.append(rnet_landmark_file)

    chose_count = assemble.assemble_data(train_file, eval_file, anno_list)

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Assemble Data')
    parser.add_argument('--root_dir', type=str, default='/datasets/WIDER', help='Dataset root')
    parser.add_argument('--anno_dir', type=str, default='anno_store', help='annotation save path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    gen_assemble_data(parse_args())
