import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ====== CONFIG ======
DATA_ROOT = os.environ.get("FL_DATA_ROOT", "/mnt/g/TEST FL PAC/clients")
NUM_CLASSES = 8
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_loaders(cid: int):
    """client_<cid> er train/val/test DataLoader."""
    base = os.path.join(DATA_ROOT, f"client_{cid}")
    train_ds = datasets.ImageFolder(os.path.join(base, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(base, "val"), eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(base, "test"), eval_tf)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **kw),
    )


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    return model


# ---- Flower <-> PyTorch parameter conversion (BatchNorm stats shoho) ----
def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, params):
    keys = model.state_dict().keys()
    state = OrderedDict((k, torch.tensor(np.asarray(v))) for k, v in zip(keys, params))
    model.load_state_dict(state, strict=True)


# ---- training / evaluation ----
def train_one_client(model, loader, epochs):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tot_loss, correct, n = 0.0, 0, 0
    for _ in range(epochs):
        tot_loss, correct, n = 0.0, 0, 0   # sesh epoch-er number report hobe
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return tot_loss / n, correct / n


@torch.no_grad()
def evaluate_model(model, loader, n_bins=15):
    """returns loss, accuracy, ECE"""
    model.to(DEVICE).eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, n = 0.0, 0
    confs, preds, labels = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(x).float()
        tot_loss += loss_fn(out, y).item()
        n += y.size(0)
        p = torch.softmax(out, dim=1)
        c, pr = p.max(dim=1)
        confs.append(c.cpu()); preds.append(pr.cpu()); labels.append(y.cpu())
    confs, preds, labels = torch.cat(confs), torch.cat(preds), torch.cat(labels)
    correct = (preds == labels).float()
    acc = correct.mean().item()

    # Expected Calibration Error
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - confs[m].mean().item())
    return tot_loss / n, acc, ece