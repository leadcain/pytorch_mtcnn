# -*- coding: utf-8 -*-

import cv2
import numpy as np

import detlib_onnx.detector.utils as utils
import onnxruntime


class DetectFace(object):
    ''' P,R,O net face detection and landmarks align '''

    def  __init__(self, pnet=None, rnet=None, onet=None, min_face=40):
        self.min_face_size = min_face   # default=12
        self.stride        = 2
        self.thresh        = [0.7, 0.7, 0.85]
        self.iou_thresh    = [0.7, 0.7, 0.7]
        self.scale_factor  = 0.709
        self._create_mtcnn_net(pnet, rnet, onet)

    def _create_mtcnn_net(self, pnet=None, rnet=None, onet=None):
        ''' Equip the basenet for detector '''
        self.pnet_detector = None
        self.rnet_detector = None
        self.onet_detector = None

        # PNet
        if pnet is not None:
            print("[INFO]: Pnet load weight {:}".format(pnet))
            self.pnet_detector = onnxruntime.InferenceSession(pnet)

        # RNet
        if rnet is not None:
            print("[INFO]: Rnet load weight {:}".format(rnet))
            self.rnet_detector = onnxruntime.InferenceSession(rnet)

        # ONet
        if onet is not None:
            print("[INFO]: Onet load weight {:}".format(onet))
            self.onet_detector = onnxruntime.InferenceSession(onet)


    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array, input image array, one batch

        Returns:
        -------
        boxes: numpy array, detected boxes before calibration
        boxes_align: numpy array, boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size   # scale = 1.0
        im_resized    = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        while current_height >= 40 and current_width >= 40:

            feed_imgs = []
            image_tensor = im_resized / 255.0
            image_tensor = np.transpose(image_tensor, (2, 0, 1)).astype(np.float32)
            feed_imgs.append(image_tensor)
            feed_imgs = np.stack(feed_imgs)

            #print("Pnet:", feed_imgs.shape)
            cls_map, reg = self.pnet_detector.run(None, {"input":feed_imgs})   # CORE， Don't look landmark

            boxes = self.generate_bbox(cls_map, reg, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], self.iou_thresh[0], mode='Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None

        all_boxes = np.vstack(all_boxes)

        keep = utils.nms(all_boxes[:, :5], 0.7, mode='Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # all_boxes = [x1, y1, x2, y2, score, reg]
        boxes_c = np.vstack([
                            all_boxes[:, 0] + all_boxes[:, 5] * bw,
                            all_boxes[:, 1] + all_boxes[:, 6] * bw,
                            all_boxes[:, 2] + all_boxes[:, 7] * bw,
                            all_boxes[:, 3] + all_boxes[:, 8] * bw,
                            all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes_c


    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array, input image array
        dets: np.array [[x1, y1, x2, y2, conf]]  
        cls_amp = [candidates, 2]
        reg     = [candidates, 4]
        Returns:
        -------
        boxes: numpy array, detected boxes before calibration
        boxes_align: numpy array, boxes after calibration
        """
        # im: an input image
        h, w, c = im.shape

        if dets is None or len(dets) == 0:
            return None
        
        #- pnet conf is not required
        dets = self.square_bbox(dets[:, :4])
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [y1, y2, x1, x2, anchor_y1, anchor_y2, \
            anchor_x1, anchor_x2, box_w, box_h] = self.boundary_check(dets, w, h)

        num_boxes = dets.shape[0]
        cropped_ims_tensors = []

        for i in range(num_boxes):
            tmp_img = np.zeros((box_h[i], box_w[i], 3), dtype=np.uint8)
            tmp_img[y1[i]:y2[i]+1, x1[i]:x2[i]+1, :] = im[anchor_y1[i]:anchor_y2[i]+1, anchor_x1[i]:anchor_x2[i]+1, :]
            crop_im = cv2.resize(tmp_img, (24, 24))
            crop_im_tensor = (crop_im / 255.0).transpose(2, 0, 1).astype(np.float32)
            cropped_ims_tensors.append(crop_im_tensor)

        feed_imgs = np.stack(cropped_ims_tensors)
        cls_map_np, reg_np = self.rnet_detector.run(None, {"input":feed_imgs}) # CORE

        #- 2d-array indicates for keeping indices
        keep_inds_np = (cls_map_np[:,1] > self.thresh[1]).nonzero()[0].reshape(-1)

        if len(keep_inds_np) > 0:
            boxes = dets[keep_inds_np]    # NOTE :: det_box from Pnet
            cls   = cls_map_np[keep_inds_np]
            reg   = reg_np[keep_inds_np]
        else:
            return None

        boxes_c = np.hstack((boxes, cls[:,1].reshape(-1,1)))
        keep = utils.nms(boxes_c, self.iou_thresh[1], mode='Minimum')

        if len(keep) == 0:
            return None
       
        keep_cls   = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg   = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_x1 = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_y1 = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_x2 = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_y2 = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_x1, align_y1, align_x2, align_y2, keep_cls[:, 1]]).T

        return boxes_align


    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array, input image array
        dets: numpy array, detection results of rnet

        Returns:
        -------
        boxes_align: numpy array, boxes after calibration
        landmarks_align: numpy array, landmarks after calibration

        """
        h, w, c = im.shape

        if dets is None or len(dets) == 0:
            return None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [y1, y2, x1, x2, anchor_y1, anchor_y2, \
            anchor_x1, anchor_x2, box_w, box_h] = self.boundary_check(dets, w, h)

        num_boxes = dets.shape[0]
        cropped_ims_tensors = []

        for i in range(num_boxes):
            tmp_img = np.zeros((box_h[i], box_w[i], 3), dtype=np.uint8)
            tmp_img[y1[i]:y2[i]+1, x1[i]:x2[i]+1, :] = im[anchor_y1[i]:anchor_y2[i]+1, anchor_x1[i]:anchor_x2[i]+1, :]
            crop_im = cv2.resize(tmp_img, (48, 48))
            crop_im_tensor = (crop_im / 255.0).transpose(2, 0, 1).astype(np.float32)
            cropped_ims_tensors.append(crop_im_tensor)

        feed_imgs = np.stack(cropped_ims_tensors)

        #print("Onet:", feed_imgs.shape)
        cls_map_np, reg_np = self.onet_detector.run(None, {"input":feed_imgs})  # look all

        keep_inds_np = (cls_map_np[:,1] > self.thresh[2]).nonzero()[0].reshape(-1)
        
        if len(keep_inds_np) > 0:
            boxes = dets[keep_inds_np]
            cls   = cls_map_np[keep_inds_np]
            reg   = reg_np[keep_inds_np]
        else:
            return None
       
        boxes_c = np.hstack((boxes, cls[:,1].reshape(-1,1)))
        keep = utils.nms(boxes_c, self.iou_thresh[2], mode="Minimum") 
 
        if len(keep) == 0:
            return None
        
        keep_cls   = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg   = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1.
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1.

        align_x1 = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_y1 = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_x2 = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_y2 = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_x1, align_y1, align_x2, align_y2, keep_cls[:, 1]]).T

        return boxes_align


    def detect_face(self, img):
        ''' Detect face over image '''
        boxes_align    = np.array([])

        # pnet
        if self.pnet_detector:
            boxes_align = self.detect_pnet(img)   # CORE
            if boxes_align is None:
                return np.array([])

        # rnet
        if self.rnet_detector:
            boxes_align = self.detect_rnet(img, boxes_align)  # CORE
            if boxes_align is None:
                return np.array([])

        # onet
        if self.onet_detector:
            boxes_align = self.detect_onet(img, boxes_align)  # CORE
            if boxes_align is None:
                return np.array([])

        return boxes_align


    def square_bbox(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """

        square_bbox = bbox.copy()
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + (w - l) * 0.5
        square_bbox[:, 1] = bbox[:, 1] + (h - l) * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1

        return square_bbox

    #
    ##- modified by bjkim
    def generate_bbox(self, score_map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            score_map : numpy array , n x 2 x m x n, detect score for each position
            reg       : numpy array , n x 4 x m x n, bbox
            scale     : float number, scale of this detection
            threshold : float number, detect threshold
        Returns:
        ----------
            bbox array
        """
        #moving 12x12 widonw with stride 2
        stride = 2
        cell_size = 12
        
        # extract positive probability and resize it as [n, m] dim
        probs = score_map[0, 1, :, :]
 
        # indices of boxes where there is probably a face
        # 2d-array indicates face candidate indices for [n, m]
        inds = np.array((probs > threshold).nonzero()).T

        if inds.shape[0] == 0:
            return np.empty((0, 9), dtype=np.int32)

        tx1, ty1, tx2, ty2 = [reg[0, i, inds[:,0], inds[:,1]] for i in range(4)]

        offsets = np.stack([tx1, ty1, tx2, ty2], 1)
        score = probs[inds[:,0], inds[:, 1]]

        bboxes = np.stack([
            np.round((stride * inds[:, 1] + 1.) / scale),
            np.round((stride * inds[:, 0] + 1.) / scale),
            np.round((stride * inds[:, 1] + 1. + cell_size) / scale),
            np.round((stride * inds[:, 0] + 1. + cell_size) / scale),
            score,
            ], 0).T
       
        bboxes = np.hstack((bboxes, offsets)).astype(np.float32)

        return bboxes

    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel, input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height  = int(height * scale)     # resized new height
        new_width   = int(width * scale)       # resized new width
        new_dim     = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def boundary_check(self, bboxes, img_w, img_h):
        """
        deal with the boundary-beyond question
        Parameters:
        ----------
            bboxes: numpy array, n x 5, input bboxes
            w: float number, width of the input image
            h: float number, height of the input image
        Returns :
        ------
            x1, y1 : numpy array, n x 1, start point of the bbox in target image
            x2, y2 : numpy array, n x 1, end point of the bbox in target image
            anchor_y1, anchor_x1 : numpy array, n x 1, start point of the bbox in original image
            anchor_x1, anchor_x2 : numpy array, n x 1, end point of the bbox in original image
            box_h, box_w         : numpy array, n x 1, height and width of the bbox
        """

        nbox = bboxes.shape[0]

        # width and height
        box_w = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        box_h = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)

        x1, y1 = np.zeros((nbox,)), np.zeros((nbox,))
        x2, y2 = box_w.copy() - 1, box_h.copy() - 1
        anchor_x1, anchor_y1 = bboxes[:, 0], bboxes[:, 1],
        anchor_x2, anchor_y2 = bboxes[:, 2], bboxes[:, 3]

        idx      = np.where(anchor_x2 > img_w - 1)
        x2[idx]  = box_w[idx] + img_w - 2 - anchor_x2[idx]
        anchor_x2[idx]  = img_w - 1

        idx      = np.where(anchor_y2 > img_h-1)
        y2[idx]  = box_h[idx] + img_h - 2 - anchor_y2[idx]
        anchor_y2[idx]  = img_h - 1

        idx     = np.where(anchor_x1 < 0)
        x1[idx] = 0 - anchor_x1[idx]
        anchor_x1[idx] = 0

        idx     = np.where(anchor_y1 < 0)
        y1[idx] = 0 - anchor_y1[idx]
        anchor_y1[idx] = 0

        return_list = [y1, y2, x1, x2, anchor_y1, anchor_y2, anchor_x1, anchor_x2, box_w, box_h]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


