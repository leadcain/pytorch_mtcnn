# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.getcwd())
from detlib.detector.imagedb import ImageDB
import detlib.train_net.train as train

anno_dir = '/datasets/WIDER_FACE/anno_store'

def train_net(args):

    imagedb    = ImageDB(args.anno_file)
    train_imdb = imagedb.load_imdb()
    #train_imdb = imagedb.append_flipped_images(train_imdb)
    
    imagedb    = ImageDB(args.eval_file)
    eval_imdb  = imagedb.load_imdb()
    
    print('train : %d\teval : %d' % (len(train_imdb), len(eval_imdb)))
    
    train.train_pnet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train PNet')

    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'pnet/train_anno_12.txt'))
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'pnet/eval_anno_12.txt'))
    parser.add_argument('--model_path', type=str,   default='./model/checkout/pnet')
    parser.add_argument('--factors',    type=list,  default=[1.0, 0.5, 0.5])
    parser.add_argument('--end_epoch',  type=int,   default=20)      
    parser.add_argument('--frequent',   type=int,   default=200)
    parser.add_argument('--lr',         type=float, default=1e-3)    
    parser.add_argument('--batch_size', type=int,   default=256)     
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--model',      type=str,   default='pnet')
    parser.add_argument('--logdir',     type=str,   default='./logs')
    parser.add_argument('--tb_write',   type=bool,  default=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
