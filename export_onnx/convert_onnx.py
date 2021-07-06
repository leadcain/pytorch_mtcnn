# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())
import argparse
import numpy as np
import torch

import detlib.detector.utils as utils
from detlib.detector.models import PNet,RNet,ONet


def export_onnx_model(model, input_shape, onnx_out_path, input_names=None, output_names=None, dynamic_axes=None):
    inputs = torch.ones(*input_shape)
    
    torch.onnx.export(model,
                      inputs,
                      onnx_out_path,
                      verbose=True,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)
    return


def main(args, device='cpu'):
    #- make save dir
    os.makedirs(args.onnx_dir, exist_ok=True)

    # PNet
    if args.pnet_weight is not None:
        pnet_detector = PNet()
        print("[INFO]: Pnet load weight {:}".format(args.pnet_weight))
        pnet_detector.load_state_dict(torch.load(args.pnet_weight, map_location=lambda storage, loc: storage))
        pnet_detector = pnet_detector.to(device)
        pnet_detector.eval()
    
        batch   = 1 
        channel = 3
        width   = 81
        height  = 60
        
        save_name = 'pnet.onnx'
        input_shape = (batch, channel, width, height)

        dynamic_axes = {'input':{0:'batch', 2:'width', 3:'height'},
                        'output':{0:'batch'}}

        export_onnx_model(pnet_detector, 
                          input_shape,
                          onnx_out_path=os.path.join(args.onnx_dir, save_name),
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)
    else:
        raise Exception("pnet weight is none...")

    # RNet
    if args.rnet_weight is not None:
        rnet_detector = RNet()
        print("[INFO]: Rnet load weight {:}".format(args.rnet_weight))
        rnet_detector.load_state_dict(torch.load(args.rnet_weight, map_location=lambda storage, loc: storage))
        rnet_detector = rnet_detector.to(device)
        rnet_detector.eval()

        batch   = 1 
        channel = 3
        width   = 24
        height  = 24
        
        save_name = 'rnet.onnx'
        input_shape = (batch, channel, width, height)

        dynamic_axes = {'input':{0:'batch'},
                        'output':{0:'batch'}}

        export_onnx_model(rnet_detector, 
                          input_shape,
                          onnx_out_path=os.path.join(args.onnx_dir, save_name),
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)
    else:
        raise Exception("pnet weight is none...")


    # ONet
    if args.onet_weight is not None:
        onet_detector = ONet()
        print("[INFO]: Onet load weight {:}".format(args.onet_weight))
        onet_detector.load_state_dict(torch.load(args.onet_weight, map_location=lambda storage, loc: storage))
        onet_detector = onet_detector.to(device)
        onet_detector.eval()

        batch   = 1 
        channel = 3
        width   = 48
        height  = 48
        
        save_name = 'onet.onnx'

        input_shape = (batch, channel, width, height)

        dynamic_axes = {'input':{0:'batch'},
                        'output':{0:'batch'}}

        export_onnx_model(onet_detector, 
                          input_shape,
                          onnx_out_path=os.path.join(args.onnx_dir, save_name),
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)


    return

def parse_args():
    parser = argparse.ArgumentParser(description='export to ONNX')
    parser.add_argument('--pnet_weight', type=str,  default=None)
    parser.add_argument('--rnet_weight', type=str,  default=None)
    parser.add_argument('--onet_weight', type=str,  default=None)
    parser.add_argument('--onnx_dir',    type=str,  default='onnx_model')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
