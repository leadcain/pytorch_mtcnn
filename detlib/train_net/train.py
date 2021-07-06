# -*- coding: utf-8 -*-

import os
import math
import time
import torch
import numpy as np
import datetime

import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import detlib.detector.image_tools as image_tools
from detlib.detector.image_reader import TrainImageReader
from detlib.detector.models import PNet,RNet,ONet
from detlib.train_net.lr_scheduler import MultiStepLR 
from detlib.train_net.loss_fn import LossFn 

from torch.utils.tensorboard import SummaryWriter


def compute_accuracy(prob_cls, gt_cls, thresh_prob = 0.6):
    ''' Just focus on negative and positive instance '''

    prob_cls = torch.squeeze(prob_cls)
    gt_cls   = torch.squeeze(gt_cls)
    prob_cls = torch.argmax(prob_cls, axis=1)

    #- only keep True of neg, pos 
    mask     = torch.ge(gt_cls, 0) # gt_cls >= 0, bool 

    label_picked = torch.masked_select(gt_cls,   mask)
    prob_picked  = torch.masked_select(prob_cls, mask)

    #boot -> int -> float for accuracy check
    equal = torch.eq(label_picked, prob_picked).int().float()
    accuracy_op = torch.mean(equal)

    return accuracy_op


def eval_net(args, net, eval_data):
    ''' Monitor the training process '''

    total = 0
    right = 0
    tp    = 0 # True positive
    fp    = 0 # False positive
    fn    = 0 # False negative
    tn    = 0 # True negative

    net.eval()
    st_acc, st_cls, st_det, st_all = 0, 0, 0, 0 
    lossfn, batch_idx = LossFn(), 1  
    for image, (gt_label, gt_bbox) in eval_data:

        im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]

        im_tensor = torch.stack(im_tensor)

        im_tensor   = Variable(im_tensor)
        gt_label    = Variable(torch.from_numpy(gt_label).float())
        gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())

        im_tensor   = im_tensor.to(args.device)
        gt_label    = gt_label.to(args.device)
        gt_bbox     = gt_bbox.to(args.device)

        with torch.no_grad():
            cls_pred, box_offset_pred = net(im_tensor)


        cls_loss        = lossfn.cls_loss(gt_label, cls_pred)
        box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
        all_loss        = cls_loss * args.factors[0] + box_offset_loss * args.factors[1] 

        accuracy = compute_accuracy(cls_pred, gt_label)
        st_acc  += accuracy.data.cpu().numpy()
        st_cls  += cls_loss.data.cpu().numpy()
        st_det  += box_offset_loss.data.cpu().numpy()
        st_all  += all_loss.data.cpu().numpy()

        batch_idx += 1

    st_acc /= batch_idx
    st_cls /= batch_idx
    st_det /= batch_idx
    st_all /= batch_idx
    st_cache = (st_acc, st_cls, st_det, st_all)
    print("Eval result acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, all_loss: %.4f" % \
          (st_acc, st_cls, st_det, st_all))

    return st_cache


def train_pnet(args, train_imdb, eval_imdb):
    workdate = datetime.datetime.now().strftime('%y%m%d').replace('\'',  '') #-yymmdd_hhmmss
    log_path = os.path.join(args.logdir, workdate, args.model)
    os.makedirs(log_path, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_path, 'tb'))

    try:
        tb_writer.add_text("train pnet")

    except:
        print("[WARN] add_text_is not work, Tensorboard version >= 2.3.0")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()
    net = PNet(is_train=True)
    net.to(args.device)

    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5) #second
    scheduler  = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

    train_data = TrainImageReader(train_imdb, 12, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb,  12, args.batch_size, shuffle=True)
    
    for cur_epoch in range(1, args.end_epoch+1):

        # training-process
        train_data.reset() # shuffle
        net.train()
        start_time = time.time()
        
        record_acc, record_cls, record_box, record_all = 0, 0, 0, 0
        for batch_idx, (image, (gt_label, gt_bbox)) in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]

            im_tensor = torch.stack(im_tensor)

            im_tensor   = Variable(im_tensor)
            gt_label    = Variable(torch.from_numpy(gt_label).float())
            gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())

            im_tensor = im_tensor.to(args.device)
            gt_label  = gt_label.to(args.device)
            gt_bbox   = gt_bbox.to(args.device)

            cls_pred, box_offset_pred = net(im_tensor)   # NOTE
            
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
            all_loss = cls_loss * args.factors[0] + box_offset_loss * args.factors[1]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()

            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, all_loss: %.4f lr: %.6f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_all/batch_idx, optimizer.param_groups[0]['lr']))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if args.tb_write:
            tags = ['train/accuracy', 'train/cls_loss', 'train/box_loss', 'train/all_loss']

            for x, tag in zip(list([accuracy, cls_loss, box_offset_loss, all_loss]), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)
        
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], cur_epoch)

        if False: #tb_histogram
            for key, param in net.state_dict().items():
                tb_writer.add_histogram(key, param.data.cpu().numpy(), cur_epoch) 

        scheduler.step()
        end_time = time.time()
        print('single epoch cost time : %.2f mins' % ((end_time-start_time) / 60))
        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)

        if args.tb_write:
            tags = ['val/accuracy', 'val/cls_loss', 'val/bbox_loss', 'val/all_loss']
        
            for x, tag in zip(list(res_cache), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)

        torch.save(net.state_dict(), os.path.join(args.model_path, "pnet_epoch_%d.pt" % cur_epoch))


def train_rnet(args, imdb, eval_imdb):
    workdate = datetime.datetime.now().strftime('%y%m%d').replace('\'',  '') #-yymmdd_hhmmss
    log_path = os.path.join(args.logdir, workdate, args.model)
    os.makedirs(log_path, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_path, 'tb'))

    try:
        tb_writer.add_text("train rnet")

    except:
        print("[WARN] add_text_is not work, Tensorboard version >= 2.3.0")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()
    net = RNet(is_train=True)
    net.to(args.device)

    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5) #second
    scheduler  = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

    train_data = TrainImageReader(imdb, args.imgsize, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb, args.imgsize, args.batch_size, shuffle=True)

    for cur_epoch in range(1, args.end_epoch+1):

        train_data.reset()
        net.train()
        record_acc, record_cls, record_box, record_all = 0, 0, 0, 0
        for batch_idx, (image, (gt_label, gt_bbox))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) \
                          for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label  = Variable(torch.from_numpy(gt_label).float())

            gt_bbox   = Variable(torch.from_numpy(gt_bbox).float())

            im_tensor = im_tensor.to(args.device)
            gt_label  = gt_label.to(args.device)
            gt_bbox   = gt_bbox.to(args.device)

            cls_pred, box_offset_pred = net(im_tensor)

            cls_loss        = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            all_loss        = cls_loss*args.factors[0] + box_offset_loss*args.factors[1]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()

            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, all_loss: %.4f lr: %.6f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_all/batch_idx, optimizer.param_groups[0]['lr']))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if args.tb_write:
            tags = ['train/accuracy', 'train/cls_loss', 'train/box_loss', 'train/all_loss']

            for x, tag in zip(list([accuracy, cls_loss, box_offset_loss, all_loss]), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)
        
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], cur_epoch)

        if False: #tb_histogram
            for key, param in net.state_dict().items():
                tb_writer.add_histogram(key, param.data.cpu().numpy(), cur_epoch) 


        scheduler.step()
        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)

        if args.tb_write:
            tags = ['val/accuracy', 'val/cls_loss', 'val/bbox_loss', 'val/all_loss']
        
            for x, tag in zip(list(res_cache), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)

        torch.save(net.state_dict(), os.path.join(args.model_path,"rnet_epoch_%d.pt" % cur_epoch))


def train_onet(args, train_imdb, eval_imdb):
    workdate = datetime.datetime.now().strftime('%y%m%d').replace('\'',  '') #-yymmdd_hhmmss
    log_path = os.path.join(args.logdir, workdate, args.model)
    os.makedirs(log_path, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_path, 'tb'))

    try:
        tb_writer.add_text("train onet")

    except:
        print("[WARN] add_text_is not work, Tensorboard version >= 2.3.0")


    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()   # TODO
    net = ONet(is_train=True)
    net.to(args.device)
    
    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5) #second
    scheduler  = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

    train_data = TrainImageReader(train_imdb, args.imgsize, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb, args.imgsize, args.batch_size, shuffle=True)

    for cur_epoch in range(1, args.end_epoch+1):

        train_data.reset()
        net.train()
        record_acc, record_cls, record_box, record_all = 0, 0, 0, 0

        for batch_idx, (image, (gt_label, gt_bbox)) in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) \
                          for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor   = Variable(im_tensor)
            gt_label    = Variable(torch.from_numpy(gt_label).float())
            gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())

            im_tensor   = im_tensor.to(args.device)
            gt_label    = gt_label.to(args.device)
            gt_bbox     = gt_bbox.to(args.device)

            cls_pred, box_offset_pred = net(im_tensor)

            cls_loss        = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)

            all_loss = cls_loss*args.factors[0] + box_offset_loss*args.factors[1]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()
                   
            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, all_loss: %.4f lr: %6f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_all/batch_idx, optimizer.param_groups[0]['lr']))
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if args.tb_write:
            tags = ['train/accuracy' 'train/cls_loss', 'train/box_loss', 'train/all_loss']

            for x, tag in zip(list([accuracy, cls_loss, box_offset_loss, all_loss]), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)
        
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], cur_epoch)

        if False: #tb_histogram
            for key, param in net.state_dict().items():
                tb_writer.add_histogram(key, param.data.cpu().numpy(), cur_epoch) 

        scheduler.step()
        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)       

        if args.tb_write:
            tags = ['val/accuracy', 'val/cls_loss', 'val/bbox_loss', 'val/all_loss']
        
            for x, tag in zip(list(res_cache), tags):
                tb_writer.add_scalar(tag, x, cur_epoch)

        torch.save(net.state_dict(), os.path.join(args.model_path, 'onet_epoch_%d.pt' % cur_epoch))
        # torch.save(net, os.path.join(args.model_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    tb_writer.close()

