# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.append(os.getcwd())

from detlib.detector.imagedb import ImageDB
import detlib.train_net.train as train
from detlib.train_net.utils import check_gpus 
from detlib.train_net.utils import cuda_benchmark


def train_net(args):
    check_gpus()
    cuda_benchmark("deterministic")

    anno_file = os.path.join(args.anno_dir, args.anno_file)
    eval_file = os.path.join(args.anno_dir, args.eval_file)

    imagedb = ImageDB(anno_file, args.root_dir)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb) # data argument

    eval_db = ImageDB(eval_file, args.root_dir)
    ev_imdb = eval_db.load_imdb()
    print('train : %d\teval : %d' % (len(gt_imdb), len(ev_imdb)))
    
    train.train_rnet(args, gt_imdb, ev_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  RNet')
    parser.add_argument('--root_dir',   type=str,   default='/datasets/WIDER')
    parser.add_argument('--anno_dir',   type=str,   default='/datasets/WIDER/anno_store')
    parser.add_argument('--anno_file',  type=str,   default='rnet/train_anno_24.txt')
    parser.add_argument('--eval_file',  type=str,   default='rnet/eval_anno_24.txt')
    parser.add_argument('--model_path', type=str,   default='model/checkout/rnet')
    parser.add_argument('--factors',    type=list,  default=[1.0, 1.0, 0.0])
    parser.add_argument('--use_lmkinfo',type=bool,  default=False)
    parser.add_argument('--imgsize',    type=int,   default=24)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=500)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=256) 
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--model',      type=str,   default='rnet')
    parser.add_argument('--logdir',     type=str,   default='./logs')
    parser.add_argument('--tb_write',   type=bool,  default=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
