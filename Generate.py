import torch
import torch.nn as nn

class CondGenerator(nn.Module):
    def __init__(self, cls_dim=384, z_dim=128, out_ch=3):
        super().__init__()
        self.latent_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cls_dim, 4*4*512),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1), nn.Tanh()
        )
    def forward(self, cls, z):
        B = cls.size(0)
        x = torch.cat([cls, z], dim=1)
        x = self.fc(x).view(B, 512, 4, 4)
        return self.net(x)
