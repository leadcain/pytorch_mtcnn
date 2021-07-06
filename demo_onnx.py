# -*- coding: utf-8 -*-

import os
import cv2
import detlib_onnx as dlib
import time
import numpy as np

from detlib_onnx.detector.detect import DetectFace

pnet_file='onnx_model/pnet.onnx'
rnet_file='onnx_model/rnet.onnx'
onet_file='onnx_model/onet.onnx'

#pnet_file=None 
#rnet_file=None
#onet_file=None

def demo_cam(w=320, h=240, min_face=24, device='cpu', interp=cv2.INTER_LINEAR):
    
    mtcnn_detector = DetectFace(
                                pnet=pnet_file,
                                rnet=rnet_file,
                                onet=onet_file,
                                min_face=min_face)

    cap    = cv2.VideoCapture(0)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print("raw image shape is width: {:} height: {:} fps: {:}".format(width, height, fps))

    #- generate canvas
    #canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        crop_img = frame[0:480, 140:500] #[h480, w360]

        #- to MTCNN
        rsimg    = cv2.resize(crop_img, dsize=(int(w), int(h)), interpolation=interp)
        #- to display
        disp_img = cv2.resize(crop_img, dsize=(720, 960), interpolation=interp)
        dist_h, dist_w, _ = disp_img.shape
        rs_h, rs_w, _ = rsimg.shape
        
        h_scale = dist_h / rs_h 
        w_scale = dist_w / rs_w

        #- bboxs[[n:, x1,x2,y1,y2,conf]] or [[]]
        stt = time.time()
        bboxs = mtcnn_detector.detect_face(rsimg)
        end = time.time()-stt

        dlib.easy_vis(disp_img, bboxs, None, scale=(w_scale, h_scale), threshold=0.95, save_path=None)

        cv2.putText(canvas, text=("[DISPLAY] {:}X{:}P@30 Det FPS[{:.2f}]".format(int(dist_w), int(dist_h), 1. / end)),
                    org=(20, 1020),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)
       
        cv2.putText(canvas, text=("[CROP] {:}X{:}".format(int(crop_img.shape[1]), int(crop_img.shape[0]))),
                    org=(760, 540),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)

        cv2.putText(canvas, text=("[RESIZE] {:}X{:}".format(int(rsimg.shape[1]), int(rsimg.shape[0]))),
                    org=(760, 870),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)


        margin = 20
        canvas[0+margin:960+margin, 0+margin:720+margin, :] = disp_img
        canvas[0+margin:480+margin, 760:760+360, :]  = crop_img
        canvas[560:560+rs_h, 760:760+rs_w, :] = rsimg

        cv2.imshow('canvas', canvas)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_dir():
    mtcnn_detector = DetectFace(
                                pnet=pnet_file,
                                rnet=rnet_file,
                                onet=onet_file,
                                min_face=12, use_cuda=False)

    for img_file in os.listdir('imgs/'):
        
        if '.jpg' in img_file:
            img = cv2.imread(os.path.join('imgs', img_file))
            bboxs, landmarks = mtcnn_detector.detect_face(img)
            save_name = 'result/r_%s' % img_file
            dlib.easy_vis(img, bboxs, landmarks, scale=(1, 1), save_path=save_name)

if __name__ == '__main__':
    #- 4:3 ratio from VGA(640, 480)
    #demo_cam(1440, 1080, min_face=24)
    #demo_cam(960, 720, min_face=24)      
    #demo_cam(640, 480, min_face=60, device='cpu')
    #demo_cam(240, 320, min_face=60, device='cpu')
    demo_cam(200, 270, min_face=40, device='cpu')
    #demo_cam(270, 360, min_face=60, device='cpu')
    #demo_cam(320, 240, min_face=60, device='cpu')
    #demo_dir()


           
