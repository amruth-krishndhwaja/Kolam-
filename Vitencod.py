import torch
import torch.nn as nn
import torchvision.models as tv

class CNNStem(nn.Module):
    def __init__(self, out_channels=384, pretrained=True):
        super().__init__()
        resnet = tv.resnet34(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.project = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        f = self.stem(x)
        p = self.project(f)
        return p

class SimpleViT(nn.Module):
    def __init__(self, in_D=384, emb_dim=384, n_blocks=6, n_heads=6, mlp_dim=1536, num_classes=4):
        super().__init__()
        self.in_D = in_D
        self.emb_dim = emb_dim
        if in_D != emb_dim:
            self.input_proj = nn.Linear(in_D, emb_dim)
        else:
            self.input_proj = None
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=mlp_dim, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim))
        self.pos_emb = None
        self.classifier = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))

    def forward(self, x):
        B, D, H, W = x.shape
        tokens = x.flatten(2).permute(0,2,1)
        if self.input_proj is not None:
            tokens = self.input_proj(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        if (self.pos_emb is None) or (self.pos_emb.shape[1] != seq.shape[1]):
            self.pos_emb = nn.Parameter(torch.randn(1, seq.shape[1], self.emb_dim).to(x.device))
        seq = seq + self.pos_emb
        seq = seq.permute(1,0,2)
        out = self.transformer(seq)
        out = out.permute(1,0,2)
        cls_out = out[:,0,:]
        logits = self.classifier(cls_out)
        return logits, cls_out

class KolamEncoder(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.stem = CNNStem(out_channels=384, pretrained=pretrained)
        self.vit = SimpleViT(in_D=384, emb_dim=384, n_blocks=6, n_heads=6, mlp_dim=1536, num_classes=num_classes)

    def forward(self, x):
        p = self.stem(x)
        logits, cls = self.vit(p)
        return logits, cls
