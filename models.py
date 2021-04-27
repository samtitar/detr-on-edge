import time
import torch
import torch.nn as nn
import torchvision.models as backbones

BACKBONES = ['resnet50', 'resnet34', 'resnet18']

class ResnetWrap(nn.Module):
    def __init__(self, backbone='resnet_50'):

        assert backbone in BACKBONES, f'Invalid backbone {backbone}, select from: {BACKBONES}'
        super().__init__()

        backbone = getattr(backbones, backbone)()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class InterpreterWrap(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 backbone='resnet50'):
        
        assert backbone in BACKBONES, f'Invalid backbone {backbone}, select from: {BACKBONES}'
        super().__init__()

        if backbone == 'resnet50':
            self.conv = nn.Conv2d(2048, hidden_dim, 1)
        elif backbone == 'resnet34' or backbone == 'resnet18':
            self.conv = nn.Conv2d(512, hidden_dim, 1)

        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_decoder_layers,
                                          num_decoder_layers)
        
        self.fc_class = nn.Linear(hidden_dim, num_classes + 1)
        self.fc_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x):
        x = self.conv(x)
        H, W = x.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        x = self.transformer(pos + 0.1 * x.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        return self.fc_class(x), self.fc_bbox(x).sigmoid()

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
        s_time = time.time()
        x = self.backbone(x)
        self.b_time = time.time() - s_time

        s_time = time.time()
        x = self.interpreter
        self.t_time = time.time() - s_time
        
        return x
