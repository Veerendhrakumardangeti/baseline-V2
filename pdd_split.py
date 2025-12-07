import numpy as np
import os

data_dir = "results/baseline"

X = np.load(f"{data_dir}/X_train.npy")
y = np.load(f"{data_dir}/y_train.npy")
d = np.load(f"{data_dir}/difficulty.npy")

idx = np.argsort(d)

N = len(X)
easy_idx = idx[:int(0.4*N)]
medium_idx = idx[:int(0.7*N)]
full_idx = idx

os.makedirs("results/pdd", exist_ok=True)

def save_split(name, idx):
    np.save(f"results/pdd/X_{name}.npy", X[idx])
    np.save(f"results/pdd/y_{name}.npy", y[idx])
    print(f"Saved {name}: {len(idx)} samples")

save_split("stage1", easy_idx)
save_split("stage2", medium_idx)
save_split("stage3", full_idx)
