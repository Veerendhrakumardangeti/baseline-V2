
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(-1)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        s = s.unsqueeze(-1)
        return x * s

class InceptionBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.b2 = nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.b3 = nn.Conv1d(in_channels, out_channels, kernel_size=39, padding=19)

        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.se = SEBlock(out_channels * 3)
        self.relu = nn.ReLU()

        if in_channels != out_channels * 3:
            self.res_conv = nn.Conv1d(in_channels, out_channels * 3, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        res = x
        a = self.b1(x)
        b = self.b2(x)
        c = self.b3(x)
        x = torch.cat([a, b, c], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        if self.res_conv is not None:
            res = self.res_conv(res)
        return self.relu(x + res)

class InceptionTimeSE(nn.Module):
    def __init__(self, in_channels, n_classes, n_blocks=6, base_channels=32):
        super().__init__()
        layers = []
        layers.append(InceptionBlockSE(in_channels, base_channels))
        in_ch = base_channels * 3
        for _ in range(1, n_blocks):
            layers.append(InceptionBlockSE(in_ch, base_channels))
        self.blocks = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def load_ckpt_model(ckpt_dir, in_channels, n_classes, device):
    """
    Robust loader: loads matching params only (shape-equal). Skips mismatched keys.
    Returns model ready for inference (unmatched layers remain at init).
    """
    model = InceptionTimeSE(in_channels=in_channels, n_classes=n_classes)

    # find newest .pth
    pths = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(pths) == 0:
        raise FileNotFoundError(f"No .pth checkpoint found inside: {ckpt_dir}")
    ckpt_path = max(pths, key=os.path.getmtime)
    print(f"[INFO] Partial-loading checkpoint: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location='cpu')

    # unwrap common dict wrappers
    if isinstance(raw, dict):
        if "state_dict" in raw:
            state = raw["state_dict"]
        elif "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw
    else:
        state = raw

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint state is not a dict; cannot load.")

    model_dict = model.state_dict()
    loaded_keys = []
    skipped_keys_shape = []
    skipped_keys_missing = []

    for k_ckpt, v_ckpt in state.items():
        # remove common prefix like module.
        k = k_ckpt
        if k.startswith("module."):
            k = k[len("module."):]

        if k in model_dict:
            model_param = model_dict[k]
            # only load if shapes exactly match
            if isinstance(v_ckpt, torch.Tensor) and v_ckpt.shape == model_param.shape:
                model_dict[k] = v_ckpt
                loaded_keys.append(k)
            else:
                skipped_keys_shape.append((k, tuple(v_ckpt.shape) if hasattr(v_ckpt, "shape") else str(type(v_ckpt)), tuple(model_param.shape)))
        else:
            skipped_keys_missing.append(k)

    # load the partially-updated state dict into model (remaining params keep init)
    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded {len(loaded_keys)} params (exact shape match).")
    if skipped_keys_shape:
        print(f"[WARN] Skipped {len(skipped_keys_shape)} keys due to shape mismatch (showing up to 10):")
        for item in skipped_keys_shape[:10]:
            print("  key:", item[0], " ckpt_shape:", item[1], " model_shape:", item[2])
    if skipped_keys_missing:
        print(f"[WARN] {len(skipped_keys_missing)} checkpoint keys did not match any model key (unexpected keys, showing up to 10):")
        for k in skipped_keys_missing[:10]:
            print("  unexpected key:", k)

    remaining_uninitialized = [k for k in model_dict.keys() if k not in loaded_keys]
    print(f"[INFO] {len(remaining_uninitialized)} model parameters were NOT loaded from checkpoint (they remain at init).")

    model.to(device)
    model.eval()
    return model


def predict(model, X, device, batch_size=64):
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            out = model(xb)
            p = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    if len(preds) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(preds), np.concatenate(probs)


def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    prec, rec, f1_vals, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    aucs = None
    if (y_prob is not None) and (y_prob.ndim == 2) and (y_prob.shape[1] > 1):
        try:
            y_onehot = np.zeros_like(y_prob)
            for i, y in enumerate(y_true):
                y_onehot[i, y] = 1
            aucs = roc_auc_score(y_onehot, y_prob, average=None)
        except Exception:
            aucs = None
    return {"acc": acc, "macro_f1": macro_f1, "prec": prec, "rec": rec, "f1": f1_vals, "cm": cm, "aucs": aucs}


def save_metrics(out_dir, prefix, m):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{prefix}_metrics.npz"), **m)
    np.savetxt(os.path.join(out_dir, f"{prefix}_confusion.csv"), m["cm"], delimiter=",")
    with open(os.path.join(out_dir, f"{prefix}_perclass.csv"), "w") as f:
        f.write("class,prec,rec,f1\n")
        for i, (p, r, f1) in enumerate(zip(m["prec"], m["rec"], m["f1"])):
            f.write(f"{i},{p},{r},{f1}\n")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))

    has_test = os.path.exists(os.path.join(args.data_dir, "X_test.npy")) and os.path.exists(os.path.join(args.data_dir, "y_test.npy"))
    if has_test:
        X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    else:
        X_test, y_test = None, None

    # ensure shape (N, C, L)
    if X_val.ndim == 2:
        X_val = X_val[:, None, :]
        if has_test:
            X_test = X_test[:, None, :]

    in_channels = X_val.shape[1]
    n_classes = len(np.unique(y_val))
    print(f"[INFO] in_channels={in_channels}, n_classes={n_classes}, device={device}")

    model = load_ckpt_model(args.ckpt_dir, in_channels, n_classes, device)

    preds_val, probs_val = predict(model, X_val, device, args.batch_size)
    if preds_val.size == 0:
        print("[WARN] No predictions produced for validation set.")
    else:
        metrics_val = compute_metrics(y_val, preds_val, probs_val)
        save_metrics(args.ckpt_dir, "val", metrics_val)
        print(f"VAL: acc={metrics_val['acc']:.4f} macro_f1={metrics_val['macro_f1']:.4f}")

    if has_test:
        preds_test, probs_test = predict(model, X_test, device, args.batch_size)
        if preds_test.size > 0:
            metrics_test = compute_metrics(y_test, preds_test, probs_test)
            save_metrics(args.ckpt_dir, "test", metrics_test)
            print(f"TEST: acc={metrics_test['acc']:.4f} macro_f1={metrics_test['macro_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
