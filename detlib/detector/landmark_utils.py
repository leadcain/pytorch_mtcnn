#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

def rotate(img, bbox, alpha):
    '''
    alpha : rotation_degree
    '''
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)

    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1], img.shape[0]))
    #crop face
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
    return (face)


def flip(face):
    ''' flip face '''
    face_flipped_by_x = cv2.flip(face, 1)
    return (face_flipped_by_x)


def randomShift(landmarkGt, shift):
    ''' Random Shift one time '''

    diff = np.random.rand(5, 2)
    diff = (2*diff - 1) * shift
    landmarkP = landmarkGt + diff
    return landmarkP


def randomShiftWithArgument(landmarkGt, shift):
    ''' Random Shift more '''

    N = 2
    landmarkPs = np.zeros((N, 6, 2))
    for i in range(N):
        landmarkPs[i] = randomShift(landmarkGt, shift)
    return landmarkPs


def is_path_exists(pathname):
    try:
      return isinstance(pathname, str) and pathname and os.path.exists(pathname)
    except OSError:
      return False


def load_txt_file(file_path):
    '''
    load data or string from text file.
    '''
    file_path = os.path.normpath(file_path)
    assert is_path_exists(file_path), 'text file is not existing!'

    with open(file_path, 'r') as file:
      data = file.read().splitlines()
    num_lines = len(data)
    file.close()

    return data, num_lines


def remove_item_from_list(list_to_remove, item):

    assert isinstance(list_to_remove, list), 'input list is not a list'
    try:
      list_to_remove.remove(item)
    except ValueError:
      print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

    return list_to_remove


def anno_parser_lmks(anno_path, num_pts = 68):
    '''
    parse the annotation for 300W dataset, which has a fixed format for .pts file
    return:
      pts: 3 x num_pts (x, y, oculusion)
    '''

    data, num_lines = load_txt_file(anno_path)
    assert data[0].find('version: ') == 0, 'version is not correct'
    assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'
    assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'

    assert data[0] == 'version: 1' or data[0] == 'version: 1.0', 'The version is wrong : {}'.format(data[0])
    n_points = int(data[1][len('n_points: '):])

    assert num_lines == n_points + 4, 'number of lines is not correct'    # 4 lines for general information: version, n_points, start and end symbol
    assert num_pts == n_points, 'number of points is not correct'

    # read points coordinate
    pts = np.zeros((n_points, 2), dtype='float32')
    line_offset = 3    # first point starts at fourth line
    point_set = set()
    for point_index in range(n_points):
      try:
        pts_list = data[point_index + line_offset].split(' ')       # x y format
        if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates
          pts_list = remove_item_from_list(pts_list, '')
        pts[point_index] = float(pts_list[0]), float(pts_list[1])
      except ValueError:
        print('error in loading points in %s' % anno_path)
    return pts, point_set

class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left   = bbox[0]
        self.top    = bbox[1]
        self.right  = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

