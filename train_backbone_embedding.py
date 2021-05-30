import time
import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import facebookresearch.datasets.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from models import ResnetWrap, InterpreterWrap

from facebookresearch.datasets.coco import CocoDetection
from facebookresearch.models.matcher import HungarianMatcher
from facebookresearch.models.detr import SetCriterion, PostProcess

from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr-drop', default=200, type=int)

    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--cos-loss', action='store_true')

    parser.add_argument('--coco-path', type=str)
    # parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--out-dir', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num-workers', default=2, type=int)

    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)

    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float)
    return parser.parse_args()


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def get_transforms():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    t_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    v_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return t_transform, v_transform

def get_criterion(num_classes, args):
    matcher = HungarianMatcher(cost_class=args.set_cost_class,
                               cost_bbox=args.set_cost_bbox,
                               cost_giou=args.set_cost_giou)
    
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}

    losses = ['labels', 'boxes', 'cardinality']
    return SetCriterion(num_classes, matcher=matcher, eos_coef=args.eos_coef,
                        weight_dict=weight_dict, losses=losses)

def main(args):
    device = args.device

    wandb.init(project='detr-on-edge')
    wandb.config.update({**{'method': 'embedding'}, **vars(args)})

    t_transform, v_transform = get_transforms()
    train_data = CocoDetection(f'{args.coco_path}/train2017', f'{args.coco_path}/annotations/instances_train2017.json', t_transform, False)
    val_data = CocoDetection(f'{args.coco_path}/val2017',  f'{args.coco_path}/annotations/instances_val2017.json', v_transform, False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    backbone_target = ResnetWrap().to(device)
    backbone = ResnetWrap(arch=args.backbone, pretrained=True).to(device)
    interpreter = InterpreterWrap(91, backbone=args.backbone).to(device)

    optimizer = optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    criterion1 = get_criterion(91, args).to(device)
    criterion2 = nn.CosineEmbeddingLoss() if args.cos_loss else nn.L1Loss()
    postprocess = PostProcess()

    detr_sd = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location='cpu', check_hash=True)

    back_sd = {k.replace('backbone.', ''): v for k, v in detr_sd.items() if k.replace('backbone.', '') in backbone_target.state_dict()}
    interp_sd = {k: v for k, v in detr_sd.items() if k in interpreter.state_dict()}
    backbone_target.load_state_dict(back_sd)
    interpreter.load_state_dict(interp_sd)

    backbone_target.eval()
    interpreter.eval()

    pbar = tqdm(total=args.epochs * len(train_loader) * args.batch_size)

    n_data, next_log = 0, 0
    for epoch in range(args.epochs):
        for samples, targets in train_loader:
            samples = [s.to(device) for s in samples]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            y_hat = backbone(samples)

            if epoch < 10:
                with torch.no_grad():
                    y_tar = backbone_target(samples)

                if args.cos_loss:
                    sim = [torch.ones(y1.shape[-1]).cuda() for y1 in y_hat]
                    loss = sum(criterion2(y1, y2, s) for y1, y2, s in zip(y_hat, y_tar, sim))
                else:
                    loss = sum(criterion2(y1, y2) for y1, y2 in zip(y_hat, y_tar))
            else:
                y_hat = interpreter(y_hat)
                y_hat = {'pred_logits': y_hat[0], 'pred_boxes': y_hat[1]}
                
                loss_dict = criterion1(y_hat, targets)
                weight_dict = criterion1.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if next_log < 1:
                if epoch < 10:
                    y_hat = interpreter(y_hat)
                    y_hat = {'pred_logits': y_hat[0], 'pred_boxes': y_hat[1]}
                
                loss_dict = criterion1(y_hat, targets)
                weight_dict = criterion1.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                step_dict = {'n_data': n_data, 'loss': loss.item()}
                wandb.log({**step_dict, **loss_dict})
                torch.save(backbone.state_dict(), f'{wandb.run.dir}/backbone_{epoch}.pt')
                next_log += 500
            
            pbar.update(args.batch_size)
            n_data += args.batch_size
            next_log -= args.batch_size
        lr_scheduler.step()

if __name__ == '__main__':
    main(get_args())