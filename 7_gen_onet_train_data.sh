#!/bin/bash

ROOT="/media/john/LIN_DATSET/WIDER"

python detlib/preprocess/gen_Onet_train_data.py \
    --root_dir ${ROOT} 
