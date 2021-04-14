from PIL import Image
from models import DETRv1

import torch
import torchvision.transforms as T

import os
import time
import requests
import argparse
import numpy as np
import matplotlib.pyplot as plt

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    img = transform(im).unsqueeze(0)

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'DETR model only supports images up to 1600 pixels on each side'

    outputs = model(img.to(DEVICE))
    probas = outputs['classes'].cpu().softmax(-1)[0, :, :-1]
    mask = probas.max(-1).values > 0.7

    bboxes_scaled = rescale_bboxes(outputs['bboxes'].cpu()[0, mask], im.size)
    return probas[mask], bboxes_scaled, outputs['info']

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()

    model = DETRv1(num_classes=91, backbone=args.backbone).to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get input format (URL, directory or filepath)
    if args.input.startswith('http'):
        images = [args.input]
    elif os.path.isdir(args.input):
        images = os.listdir(args.input)
        images = [os.path.join(args.input, x) for x in images]
    elif os.path.isfile(args.input):
        images = [args.input]
    else:
        raise ValueError(f'Invalid input {args.input}')
    
    # Process each image
    b_times, t_times = [], []
    for i, image in enumerate(images):
        if image.startswith('http'):
            image = requests.get(image, stream=True).raw
        img = Image.open(image)

        try:
            proba, bbox, info = detect(img, model, transform)
            b_time, t_time = info['b_time'], info['t_time']
            t_total = b_time + t_time

            print(f't_backbone: {100 * b_time / t_total:.2f}%\t t_transformer: {100 * t_time / t_total:.2f}%\t t_total: {t_total:.2f}')
            b_times.append(b_time / t_total)
            t_times.append(t_time / t_total)
        except Exception as e:
            continue

        # Plot output and store if necessary
        if args.output == 'none':
            continue
        elif args.output == 'plot':
            plot_results(img, proba, bbox)
            plt.show()
        elif os.path.isdir(args.output):
            plot_results(img, proba, bbox)
            path = os.path.join(args.output, f'{i}.jpg')
            plt.savefig(path)
        else:
            raise ValueError(f'Invalid output {args.output}')

    if len(b_times) > 0:
        print(f'Average backbone time: {np.mean(b_times)}')
        print(f'Average transformer time: {np.mean(t_times)}')
    else:
        print('No images processed, is the input directory empty?')