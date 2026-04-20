"""
Train tomato leaf disease classifier.
Usage: python train.py --data_dir data/tomato --model efficientnet_b3
"""
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import build_model

CLASSES = ["BacterialSpot","EarlyBlight","LateBlight","LeafMold",
           "SeptoriaLeafSpot","SpiderMites","TargetSpot",
           "MosaicVirus","YellowLeafCurlVirus","Healthy"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/tomato")
    p.add_argument("--model",      default="efficientnet_b3")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tfm = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.1), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        "val": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    }
    loaders = {s: DataLoader(datasets.ImageFolder(f"{args.data_dir}/{s}", tfm[s]),
               args.batch_size, shuffle=(s=="train"), num_workers=4, pin_memory=True)
               for s in ("train","val")}

    model     = build_model(args.model, num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        for phase in ("train","val"):
            model.train() if phase=="train" else model.eval()
            total_loss = correct = total = 0
            for imgs, labels in loaders[phase]:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.set_grad_enabled(phase=="train"):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    if phase=="train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1)==labels).sum().item()
                total   += len(labels)
            acc = correct/total
            if phase=="val":
                print(f"Epoch {epoch:3d} | val_acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
                    print(f"  ✅ Best acc={best_acc:.4f}")
        scheduler.step()

if __name__=="__main__": main()
