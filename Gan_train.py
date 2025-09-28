import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.generator import CondGenerator
from models.discriminator import PatchDiscriminator
from models.encoder import KolamEncoder

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

def train(args):
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    ds = ImageDataset(args.data, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = KolamEncoder(num_classes=4, pretrained=True).to(device)
    if args.encoder and os.path.exists(args.encoder):
        encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    encoder.eval()
    G = CondGenerator(cls_dim=384, z_dim=128, out_ch=3).to(device)
    D = PatchDiscriminator(in_ch=3, cls_dim=384).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    adversarial_loss = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        for imgs in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                _, cls = encoder(imgs)
            B = imgs.size(0)
            real_labels = torch.ones(B,1,1,1).to(device)
            fake_labels = torch.zeros(B,1,1,1).to(device)
            # train D
            z = torch.randn(B, 128).to(device)
            fake = G(cls, z)
            D_real = D(imgs, cls)
            D_fake = D(fake.detach(), cls)
            lossD = (adversarial_loss(D_real, real_labels) + adversarial_loss(D_fake, fake_labels)) * 0.5
            optD.zero_grad(); lossD.backward(); optD.step()
            # train G
            D_fake_forG = D(fake, cls)
            lossG = adversarial_loss(D_fake_forG, real_labels)
            optG.zero_grad(); lossG.backward(); optG.step()
        print(f"Epoch {epoch+1}/{args.epochs} generator_loss={lossG.item():.4f} disc_loss={lossD.item():.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(G.state_dict(), args.out)
    print("Saved generator to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--encoder", default="")
    parser.add_argument("--out", default="models/generator.pth")
    args = parser.parse_args()
    train(args)
