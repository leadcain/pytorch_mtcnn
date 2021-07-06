#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random

class ImageDB(object):

    def __init__(self, image_annotation_file, root_dir, prefix_path='', mode='train'):
        self.image_annotation_file = image_annotation_file
        self.root_dir = root_dir
        self.prefix_path = prefix_path
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.mode = mode


    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def load_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """

        gt_imdb = self.load_annotations()

        return gt_imdb


    def real_image_path(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")
       
        #print("DBG:", index)
 
        if not os.path.exists(index):
            image_file = os.path.join(self.root_dir, 'face_detection/WIDER_train/images', self.prefix_path, index)
        else:
            image_file=index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        #print("DBG:", image_file)
        return image_file


    def load_annotations(self, annotion_type=1):
        """Load annotations

        Parameters:
        ----------
        annotion_type: int
                      0:dsadsa
                      1:dsadsa
        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]

            im_path = self.real_image_path(index)  # BUG  
            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
                pass
            else:
                im_path = self.real_image_path(index)  # BUG  
                imdb_ = dict()
                imdb_['image'] = im_path

                label = annotation[1]
                imdb_['label']           = int(label)
                imdb_['flipped']         = False
                imdb_['bbox_target']     = np.zeros((4,))

                if len(annotation[2:])==4: #only bboxes exist
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)

            imdb.append(imdb_)

        return imdb


    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('Flipping action will add %d images to imdb' % len(imdb))
        for i in range(len(imdb)):
            imdb_  = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0] # mirror-filp, but coord is kept

            item = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'flipped': True}

            imdb.append(item)
        self.image_set_index *= 2
        return imdb

