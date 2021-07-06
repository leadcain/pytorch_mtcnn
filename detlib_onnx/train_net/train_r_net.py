# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.append(os.getcwd())

from detlib.detector.imagedb import ImageDB
import detlib.train_net.train as train

anno_dir = '/datasets/WIDER_FACE/anno_store'


def train_net(args):

    imagedb = ImageDB(args.anno_file)
    gt_imdb = imagedb.load_imdb()
    #gt_imdb = imagedb.append_flipped_images(gt_imdb) # data argument

    eval_db = ImageDB(args.eval_file)
    ev_imdb = eval_db.load_imdb()
    print('train : %d\teval : %d' % (len(gt_imdb), len(ev_imdb)))
    
    train.train_rnet(args, gt_imdb, ev_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  RNet')

    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'rnet/train_anno_24.txt'))
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'rnet/eval_anno_24.txt'))
    parser.add_argument('--model_path', type=str,   default='model/checkout/rnet')
    parser.add_argument('--factors',    type=list,  default=[1.0, 1.0, 0.5])
    parser.add_argument('--use_lmkinfo',type=bool,  default=False)       # TODO
    parser.add_argument('--imgsize',    type=int,   default=24)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=500)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=256)     # default = 32
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--model',      type=str,   default='rnet')
    parser.add_argument('--logdir',     type=str,   default='./logs')
    parser.add_argument('--tb_write',   type=bool,  default=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
