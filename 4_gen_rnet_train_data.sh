#!/bin/bash

ROOT="/media/john/LIN_DATSET/WIDER"
DEVICE="cuda:0"

python detlib/preprocess/gen_Rnet_train_data.py \
    --root_dir ${ROOT} \
    --device ${DEVICE}
