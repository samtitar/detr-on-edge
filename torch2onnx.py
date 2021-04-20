from PIL import Image
from models import DETRv1

import torch
import torchvision
import torchvision.transforms as T

import os
import onnx
import requests
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()

    model = DETRv1(num_classes=91, backbone=args.backbone)
    model.eval().cuda()

    in_tensor = torch.zeros((1, 3, 512, 512)).cuda()
    
    dynamic_axes = {'inputs': {0: 'batch', 2: 'width', 3: 'height'},
                    'outputs': {0: 'batch'}}
    
    torch.onnx.export(model, in_tensor, args.output,
                      input_names=['inputs'], output_names=['outputs'],
                      dynamic_axes=dynamic_axes, opset_version=10)
    