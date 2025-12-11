import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset
# -------------------------
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


# -------------------------
# InceptionTime (strong baseline)
# -------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.b2 = nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.b3 = nn.Conv1d(in_channels, out_channels, kernel_size=39, padding=19)
        self.bn = nn.BatchNorm1d(out_channels * 3)

    def forward(self, x):
        c1 = self.b1(x)
        c2 = self.b2(x)
        c3 = self.b3(x)
        out = torch.cat([c1, c2, c3], dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(self, in_channels, n_classes):
        """
        in_channels: number of input channels (e.g., 12 for 12-lead ECG)
        """
        super().__init__()
        # block1 will expand channels -> out channels = 32 * 3 = 96
        self.block1 = InceptionBlock(in_channels, out_channels=32)
        # subsequent blocks take 96 input channels (32*3)
        self.block2 = InceptionBlock(96, out_channels=32)
        self.block3 = InceptionBlock(96, out_channels=32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# -------------------------
# Training / Evaluation
# -------------------------
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    count = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # --- DEBUG PRINT: show batch shape once per epoch ---
        if i == 0:
            try:
                print(f"[DEBUG] batch {i} — x.shape: {x.shape}, x.dtype: {x.dtype}, device: {x.device}")
            except Exception as _err:
                print(f"[DEBUG] batch {i} — unable to print x.shape: {_err}")
        # ----------------------------------------------------

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        count += x.size(0)

    return total_loss / count if count > 0 else 0.0, (correct / count) if count > 0 else 0.0


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            count += x.size(0)

    return total_loss / count if count > 0 else 0.0, (correct / count) if count > 0 else 0.0


# -------------------------
# Progressive Data Dropout (PDD-SRD)
# -------------------------
def pdd_select_indices(difficulty, easy_pct, medium_pct, epoch, total_epochs, gamma):
    # Phase thresholds
    easy_threshold = np.percentile(difficulty, easy_pct * 100)
    medium_threshold = np.percentile(difficulty, medium_pct * 100)

    if epoch < total_epochs * 0.5:
        # Hard-sample focus: keep samples >= easy_threshold
        keep = np.where(difficulty >= easy_threshold)[0]
    else:
        # Reintroduce medium samples
        keep = np.where(difficulty >= medium_threshold)[0]

    # Soft reintroduction (gamma decay)
    prob = gamma ** epoch
    if len(keep) == 0:
        return keep  # empty array
    mask = np.random.rand(len(keep)) < prob
    return keep[mask]


# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    # Load baseline split
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))

    difficulty = np.load(args.difficulty)
    n_classes = len(np.unique(y_train))

    # Ensure arrays are shape (N, C, L)
    if X_train.ndim == 2:
        # (N, L) -> (N, 1, L)
        X_train = X_train[:, None, :]
        X_val = X_val[:, None, :]
    elif X_train.ndim == 3:
        # (N, C, L) — fine
        pass
    else:
        raise ValueError(f"Unexpected X_train ndim: {X_train.ndim}. Expected 2 or 3 dims.")

    in_channels = X_train.shape[1]
    print(f"[INFO] Detected input channels: {in_channels}. Signal length: {X_train.shape[2]}")

    model = InceptionTime(in_channels, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # DataLoader kwargs
    dl_kwargs = {"batch_size": args.batch_size, "shuffle": True, "num_workers": args.num_workers}
    val_dl_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers}
    if use_cuda:
        dl_kwargs["pin_memory"] = True
        val_dl_kwargs["pin_memory"] = True

    for epoch in range(1, args.epochs + 1):
        indices = pdd_select_indices(
            difficulty,
            args.easy_pct,
            args.medium_pct,
            epoch,
            args.epochs,
            args.gamma
        )

        # If pdd_select_indices returned empty, fall back to using all training samples
        if len(indices) == 0:
            print(f"[WARN] PDD selected 0 indices for epoch {epoch}. Falling back to all training samples.")
            indices = np.arange(len(X_train))

        print(f"Epoch {epoch}: using {len(indices)} / {len(X_train)} samples")

        train_ds = ECGDataset(X_train[indices], y_train[indices])
        train_loader = DataLoader(train_ds, **dl_kwargs)

        val_ds = ECGDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, **val_dl_kwargs)

        train_loss, train_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{args.epochs}  Train Loss={train_loss:.4f} Acc={train_acc:.4f}  Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
            print("Saved best model\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--difficulty", type=str, required=True)
    p.add_argument("--model", type=str, default="inception")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--easy_pct", type=float, default=0.4)
    p.add_argument("--medium_pct", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()
    main(args)
