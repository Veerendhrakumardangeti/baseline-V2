import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, X, y):
        if X.ndim == 2:
            X = X[:, None, :]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        b1 = out_channels // 4
        self.p1 = nn.Conv1d(in_channels, b1, kernel_size=1)
        self.p2 = nn.Conv1d(in_channels, b1, kernel_size=3, padding=1)
        self.p3 = nn.Conv1d(in_channels, b1, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.p4 = nn.Conv1d(in_channels, b1, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(self.pool(x))], dim=1)
        return self.relu(self.bn(out))

class InceptionTime(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.b1 = InceptionBlock(in_channels, 64)
        self.b2 = InceptionBlock(64, 64)
        self.b3 = InceptionBlock(64, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    in_channels = 1 if X_train.ndim == 2 else X_train.shape[1]
    n_classes = int(y_train.max() + 1)

    train_ds = ECGDataset(X_train, y_train)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = InceptionTime(in_channels, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 1e9
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_inception.pth"))

    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best_inception.pth")))
    te_loss, te_acc = eval_epoch(model, test_loader, loss_fn, device)
    print("Test Loss:", te_loss)
    print("Test Accuracy:", te_acc)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()
    main(args)
