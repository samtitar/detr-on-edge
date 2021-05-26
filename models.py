import time
import torch
import torch.nn as nn
import torchvision.models as backbones

BACKBONES = ['resnet50', 'resnet34', 'resnet18']

class ResnetWrap(nn.Module):
    def __init__(self, hidden_dim=256, arch='resnet50', pretrained=False):

        assert arch in BACKBONES, f'Invalid architecture {arch}, select from: {BACKBONES}'
        super().__init__()

        backbone = getattr(backbones, arch)(pretrained=pretrained)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if arch == 'resnet50':
            self.conv2 = nn.Conv2d(2048, hidden_dim, 1)
        elif arch == 'resnet34' or arch == 'resnet18':
            self.conv2 = nn.Conv2d(512, hidden_dim, 1)
    
    def forward(self, x):
        if isinstance(x, list):
            return [self(e.unsqueeze(0)).squeeze(0) for e in iter(x)]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        
        return x

class InterpreterWrap(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 backbone='resnet50'):
        
        assert backbone in BACKBONES, f'Invalid backbone {backbone}, select from: {BACKBONES}'
        super().__init__()

        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers,
                                          num_decoder_layers)
        
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x):
        if isinstance(x, list):
            cs, bs = tuple(zip(*[self(e.unsqueeze(0)) for e in x]))
            return torch.cat(cs), torch.cat(bs)

        H, W = x.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        x = self.transformer(pos + 0.1 * x.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        return self.linear_class(x), self.linear_bbox(x).sigmoid()

class DETRv1(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 backbone='resnet50'):
        
        assert backbone in BACKBONES, f'Invalid backbone {backbone}, select from: {BACKBONES}'
        super().__init__()

        self.backbone = ResnetWrap(backbone)
        self.interpreter = InterpreterWrap(num_classes, hidden_dim, nheads,
                                           num_encoder_layers, num_decoder_layers,
                                           backbone)

    def forward(self, x):
        x = self.backbone(x)
        x = self.interpreter(x)
        return x
