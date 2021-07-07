# pytorch_mtcnn
trainable MTCNN with some modification(loss function, etc.)  

# Environment
python, pytorch, numpy etc.
for detail library dependency please cneck the requirement.yaml  

# Preparation
 WIDER FACE for face detection
 > http://shuoyang1213.me/WIDERFACE/
 
 
 CelebA for landmark for face landmark detection 
 > https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  

# Training:
 source bash file by oder
 >> 1_gen_pnet_train_data.sh \
 >> 2_assmble_pnet_imglist.sh \
 >> 3_train_pnet.sh \
 >> ...
 
# ONNX Conversion
> source 98_convert_onnx.sh
 
# Demo on Raspberry Pi4
  FPS : about 20~40 FPS on 320X240(CPU only)
  * RB Pi4 environment:
     - ubuntu 20.04 LTS
     - miniconda
     - onnxruntime
     - python
     - numpy
     - opencv
 
 
# NOTE
 1. this version is not use landmark layers but you can add for landmark detection with celebA dataset
 2. accept ohem loss function
 3. some modification
 

# Reference 
> https://github.com/Ontheway361/mtcnn-pytorch
> https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf
> https://paperswithcode.com/dataset/wider-face-1
