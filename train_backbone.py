import wandb
import argparse

import torch
import torch.optim as optim
import facebookresearch.datasets.transforms as T

from torch.utils.data import DataLoader
from models import ResnetWrap, InterpreterWrap

from facebookresearch.datasets.coco import CocoDetection
from facebookresearch.models.matcher import HungarianMatcher
from facebookresearch.models.detr import SetCriterion, PostProcess

def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr-drop', default=200, type=int)

    parser.add_argument('--backbone', type=str, default='resnet50')

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
    wandb.config.update(args)

    t_transform, v_transform = get_transforms()
    train_data = CocoDetection(f'{args.coco_path}/train2017', f'{args.coco_path}/annotations/instances_train2017.json', t_transform, False)
    val_data = CocoDetection(f'{args.coco_path}/val2017',  f'{args.coco_path}/annotations/instances_val2017.json', v_transform, False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    backbone = ResnetWrap(arch=args.backbone, pretrained=True).to(device)
    interpreter = InterpreterWrap(91, backbone=args.backbone).to(device)
    for p in interpreter.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    criterion = get_criterion(91, args).to(device)
    postprocessors = {'bbox': PostProcess()}

    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location='cpu', check_hash=True)
    
    state_dict = {k: v for k, v in state_dict.items() if k in interpreter.state_dict()}
    interpreter.load_state_dict(state_dict)

    n_data, next_log = 0, 0
    for epoch in range(args.epochs):
        for samples, targets in train_loader:
            samples = [s.to(device) for s in samples]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            y_hat = backbone(samples)
            y_hat = interpreter(y_hat)
            y_hat = {'pred_logits': y_hat[0], 'pred_boxes': y_hat[1]}

            loss_dict = criterion(y_hat, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if next_log < 1:
                step_dict = {'n_data': n_data, 'loss': loss.item()}
                wandb.log({**step_dict, **loss_dict})
                next_log += 500

            n_data += args.batch_size
            next_log -= args.batch_size
        lr_scheduler.step()

if __name__ == '__main__':
    main(get_args())