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

    imagedb    = ImageDB(anno_file, args.root_dir)
    train_imdb = imagedb.load_imdb()
    train_imdb = imagedb.append_flipped_images(train_imdb)
    
    imagedb    = ImageDB(eval_file, args.root_dir)
    eval_imdb  = imagedb.load_imdb()
    
    print('train : %d\teval : %d' % (len(train_imdb), len(eval_imdb)))
    
    train.train_onet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  ONet')
    parser.add_argument('--root_dir',   type=str,   default='/datasets/WIDER')
    parser.add_argument('--anno_dir',   type=str,   default='/datasets/WIDER/anno_store')
    parser.add_argument('--anno_file',  type=str,   default='onet/train_anno_48.txt')
    parser.add_argument('--eval_file',  type=str,   default='onet/eval_anno_48.txt')
    parser.add_argument('--model_path', type=str,   default='model/checkout/onet')
    parser.add_argument('--factors',    type=list,  default=[1., 1., 0.])  
    parser.add_argument('--use_lmkinfo',type=bool,  default=False)
    parser.add_argument('--imgsize',    type=int,   default=48)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=1e-3)    
    parser.add_argument('--batch_size', type=int,   default=256)     
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--model',      type=str,   default='onet')
    parser.add_argument('--tb_write',   type=bool,  default=True)
    parser.add_argument('--logdir',     type=str,   default='./logs')
    parser.add_argument('--prefix_path',type=str,   default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
