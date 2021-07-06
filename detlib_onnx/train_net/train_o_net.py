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
    
    train.train_onet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  ONet')
    
    # {landmark_48_train:311617, landmark_48_cls_train:15809}
    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'onet/train_anno_48.txt'))  
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'onet/eval_anno_48.txt'))
    parser.add_argument('--model_path', type=str,   default='model/checkout/onet')
    parser.add_argument('--factors',    type=list,  default=[1., 1., 1.])  
    parser.add_argument('--use_lmkinfo',type=bool,  default=True)
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
