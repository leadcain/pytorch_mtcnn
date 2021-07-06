#!/bin/bash

ROOT="/media/john/LIN_DATSET/WIDER"

python detlib/preprocess/assemble_rnet_imglist.py \
    --root_dir ${ROOT}
