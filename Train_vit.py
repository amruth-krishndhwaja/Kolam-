import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.encoder import KolamEncoder

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.files = []
        for cls in os.listdir(root_dir):
            p = os.path.join(root_dir, cls)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        self.files.append((os.path.join(p,f), cls))
        self.classes = sorted(list({c for _,c in self.files}))
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path, cls = self.files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.cls2idx[cls]
        return img, label

def train(args):
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor()
    ])
    ds = ImageFolderDataset(args.data, transform=transform)
    if len(ds.classes) == 0:
        raise RuntimeError("No labeled classes found in data directory (expect class subfolders).")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KolamEncoder(num_classes=len(ds.classes), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss=0
        total=0
        correct=0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, _ = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(loader):.4f} acc={correct/total:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("Saved encoder to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dir with class subfolders of images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default="models/encoder.pth")
    args = parser.parse_args()
    train(args)
