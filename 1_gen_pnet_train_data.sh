#!/bin/bash

ROOT="/media/john/LIN_DATSET/WIDER"

python detlib/preprocess/gen_Pnet_train_data.py \
    --root_dir ${ROOT} \
    --num_neg 60 \
    --num_pos 30 \
    --num_rot 5 \
    --iou_pos 0.65 \
    --iou_neg 0.25 \
    --iou_part 0.4 
