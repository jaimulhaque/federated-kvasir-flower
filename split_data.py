import random
import shutil
from pathlib import Path

# ====== CONFIG ======
RAW_DIR = Path("/mnt/g/TEST FL PAC/RAW DATASET/kvasir-dataset-v2")
OUT_DIR = Path("/mnt/g/TEST FL PAC/clients")
NUM_CLIENTS = 4
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # baki 0.15 = test
SEED = 42
# ====================

random.seed(SEED)
classes = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
print("Classes:", classes)

# prottek client-er jonno counter (report-er jonno)
counts = {c: {s: 0 for s in ("train", "val", "test")} for c in range(1, NUM_CLIENTS + 1)}

for cls in classes:
    files = sorted(p for p in (RAW_DIR / cls).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    random.shuffle(files)
    per_client = len(files) // NUM_CLIENTS   # 1000 / 4 = 250

    for i in range(NUM_CLIENTS):
        cid = i + 1
        chunk = files[i * per_client:(i + 1) * per_client]
        n_train = int(len(chunk) * TRAIN_FRAC)
        n_val = int(len(chunk) * VAL_FRAC)
        splits = {
            "train": chunk[:n_train],
            "val": chunk[n_train:n_train + n_val],
            "test": chunk[n_train + n_val:],
        }
        for split, items in splits.items():
            dest = OUT_DIR / f"client_{cid}" / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dest / f.name)
            counts[cid][split] += len(items)

print("\nDone. Per-client totals:")
for cid, c in counts.items():
    print(f"client_{cid}: train={c['train']} val={c['val']} test={c['test']} "
          f"total={sum(c.values())}")