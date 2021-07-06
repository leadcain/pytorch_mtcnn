# pytorch_mtcnn
pytorch mtcnn

Implementation of pytorch mtcnn with trainable on PYTORCH > 1.3

environment : check the 'requirement.yaml'

Download dataset :
 WIDER FACE for face detection :  http://shuoyang1213.me/WIDERFACE/
 
 CelebA for landmark for face landmark detection :  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 
Wider face challenge : 
  https://paperswithcode.com/dataset/wider-face-1

Train step:
 following bash file by oder
 
 
ONNX conversion
source 98_convert_onnx.sh

Demo on Raspberry Pi4
  about 20~40 FPS on 320X240
  * RB Pi4 environment:
      miniconda, onnxruntime, python, numpy only
 
 
NOTE:
 this version is not use landmark layers but you can add for landmark detection
 

Reference 
https://github.com/Ontheway361/mtcnn-pytorch
https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf
