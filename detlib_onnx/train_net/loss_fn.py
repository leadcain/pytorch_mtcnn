# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class LossFn:
    '''
        gt_label : neg=0, pos=1, part=-1, landmark=-2
    '''
    def __init__(self, cls_factor = 1, box_factor = 1, landmark_factor = 1):

        self.cls_factor  = cls_factor
        self.box_factor  = box_factor
        self.land_factor = landmark_factor

        # make the first 70% of the mini-batch difficult samples
        self.num_keep_ratio = 0.7

    #- class-OHEM loss added by bjkim
    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label   = torch.squeeze(gt_label)

        zeros = torch.zeros(gt_label.shape).cuda()

        #- pos=1, neg=0, others=0
        label_filter_invalid = torch.where(gt_label < 0., zeros, gt_label)

        #- get number of pred_label
        cls_prob_reshape = torch.reshape(pred_label, [torch.numel(pred_label), -1])

        # get the number of rows of pred_label
        num_row = torch.numel(pred_label)
        row = torch.arange(0, num_row, 2) # [0, 2, 4,...]

        indices_ = row.cuda() + label_filter_invalid.type(torch.int)
        label_prob = torch.squeeze(torch.index_select(cls_prob_reshape, 0, indices_))
        
        loss = -torch.log(label_prob + 1e-10)

        zeros = torch.zeros(label_prob.shape, dtype=torch.float).cuda()
        ones  = torch.ones(label_prob.shape, dtype=torch.float).cuda()

        # set pos & neg to be 1, others to be 0        
        valid_inds = torch.where(gt_label < zeros, zeros, ones)

        # get the number of POS and NEG examples
        num_valid = torch.sum(valid_inds)
        keep_num = num_valid * self.num_keep_ratio

        loss = loss * valid_inds
        loss, _ = torch.topk(loss, k=keep_num.type(torch.int))
        loss =  torch.mean(loss) * self.cls_factor

        #print("cls loss:", loss)
        return loss


    #- box_ohem_loss added by bjkim
    def box_loss(self, gt_label, gt_offset, pred_offset):
        '''
            gt_label    : 1d-array indicates neg, pos, part, landmark 
            gt_offset   : 2d-array indicates gt bbox cordinate
            pred_offset : 2d-array indicates pred bbox cordinate
            return : mean euclidian loss for all the pos & part examples
        '''
        pred_offset = torch.squeeze(pred_offset) #pred box
        gt_offset   = torch.squeeze(gt_offset)   #gt box
        gt_label    = torch.squeeze(gt_label)    #gt_label : pos, neg, part, landmark

        zeros = torch.zeros(gt_label.shape, dtype=torch.float).cuda() 
        ones  = torch.ones(gt_label.shape,  dtype=torch.float).cuda()

        #- keep pos & part example
        valid_inds = torch.where(torch.eq(torch.abs(gt_label), 1), ones, zeros)

        #- calculate square sum
        square_error = torch.square(pred_offset-gt_offset)
        square_error = torch.sum(square_error, axis=1)
      
        #- get the number of valid 
        num_valid = torch.sum(valid_inds)

        #- get square errors of valid index
        square_error = square_error * valid_inds

        #- keep top-k examples, k equals to the number of positive examples 
        _, k_index = torch.topk(square_error, k=num_valid.type(torch.int))
        square_error = torch.gather(square_error, 0, k_index)

        loss = torch.mean(square_error) * self.box_factor
        #print("box loss:", loss)
        return loss


    #- lanbdmark_ohem_loss added by bjkim
    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        '''
            gt_label      : 1d-array indicates neg, pos, part, landmark 
            gt_landmark   : 1d-array indicates landmark points
            pred_landmark : 1d-array indicates pred landmark points
            return : mean euclidian loss for all the pos & part examples
        '''
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark   = torch.squeeze(gt_landmark)
        gt_label      = torch.squeeze(gt_label)

        zeros = torch.zeros(gt_label.shape, dtype=torch.float).cuda() 
        ones  = torch.ones(gt_label.shape,  dtype=torch.float).cuda()

        valid_inds = torch.where(torch.eq(gt_label, -2), ones, zeros)

        square_error = torch.square(pred_landmark-gt_landmark)
        square_error = torch.sum(square_error, axis=1)

        num_valid = torch.sum(valid_inds)

        #- when calculating landmark_ohem loss, only calculate beta=1
        square_error = square_error * valid_inds

        _, k_index = torch.topk(square_error, k=num_valid.type(torch.int))
        square_error = torch.index_select(square_error, 0, k_index)
        
        loss = torch.mean(square_error) * self.land_factor 
     
        #print("lmk loss:", loss)
        return loss 

