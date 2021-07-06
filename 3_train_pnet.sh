#!/bin/bash

ROOT="/media/john/LIN_DATSET/WIDER"
DEVICE="cuda:0"

python detlib/train_net/train_p_net.py  \
    --root_dir ${ROOT} \
    --anno_dir ${ROOT}/anno_store \
    --device ${DEVICE}
