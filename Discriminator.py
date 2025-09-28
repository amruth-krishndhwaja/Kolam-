import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, cls_dim=384):
        super().__init__()
        self.c_proj = nn.Linear(cls_dim, 64)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch+1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0)
        )
    def forward(self, img, cls):
        B, C, H, W = img.shape
        c = self.c_proj(cls).unsqueeze(-1).unsqueeze(-1)
        c_map = c.expand(-1, -1, H, W)[:, :1, :, :]
        x = torch.cat([img, c_map], dim=1)
        out = self.net(x)
        return out
