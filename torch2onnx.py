from PIL import Image
from numpy import interp
from models import DETRv1, ResnetWrap, InterpreterWrap

import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()

    in_tensor = torch.zeros((1, 3, 224, 224))
    if args.split == False:
        model = DETRv1(91, backbone=args.backbone).eval()
        
        dynamic_axes = {'image': {0: 'batch'},
                        'class': {0: 'batch'}, 'bbox': {0: 'batch'}}

        torch.onnx.export(model, in_tensor, f'{args.outdir}/detr.onnx',
                          input_names=['image'], output_names=['class', 'bbox'],
                          dynamic_axes=dynamic_axes, opset_version=10)
    
    if args.split == True:
        backbone = ResnetWrap(args.backbone).eval()
        interpreter = InterpreterWrap(91).eval()

        dynamic_axes = {'image': {0: 'batch'},
                        'vector': {0: 'batch'}}

        torch.onnx.export(backbone, in_tensor, f'{args.outdir}/backbone.onnx',
                          input_names=['image'], output_names=['vector'],
                          dynamic_axes=dynamic_axes, opset_version=10)
        in_tensor = backbone(in_tensor) # Dims depend on resnet model

        dynamic_axes = {'vector': {0: 'batch'},
                        'class': {0: 'batch'}, 'bbox': {0: 'batch'}}

        torch.onnx.export(interpreter, in_tensor, f'{args.outdir}/interpreter.onnx',
                          input_names=['vector'], output_names=['class', 'bbox'],
                          dynamic_axes=dynamic_axes, opset_version=10)
        a = interpreter(in_tensor)
