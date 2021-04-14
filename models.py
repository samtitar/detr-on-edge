import time
import torch
import torch.nn as nn
import torchvision.models as backbones

BACKBONES = ['resnet50', 'resnet34', 'resnet18']

class DETRv1(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 backbone='resnet50'):
        
        assert backbone in BACKBONES, f'Invalid backbone {backbone}, select from: {BACKBONES}'

        super().__init__()
        self.backbone = getattr(backbones, backbone)()
        del self.backbone.fc
        
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
        s_time = time.time()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        b_time = time.time() - s_time

        s_time = time.time()
        x = self.conv(x)
        H, W = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        x = self.transformer(pos + 0.1 * x.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        t_time = time.time() - s_time

        return {'classes': self.fc_class(x),
                'bboxes': self.fc_bbox(x).sigmoid(),
                'info': {'b_time': b_time, 't_time': t_time}}
