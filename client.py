import argparse
import fcntl
import time
from contextlib import contextmanager

import flwr as fl
import torch

from common import (DEVICE, build_model, evaluate_model, get_loaders,
                    get_parameters, set_parameters, train_one_client)

GPU_LOCK_FILE = "/tmp/fl_gpu.lock"


@contextmanager
def gpu_lock():
    with open(GPU_LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, epochs: int):
        self.cid = cid
        self.epochs = epochs
        self.model = build_model()
        self.train_loader, self.val_loader, self.test_loader = get_loaders(cid)
        print(f"[client {cid}] train={len(self.train_loader.dataset)} "
              f"val={len(self.val_loader.dataset)} test={len(self.test_loader.dataset)} "
              f"device={DEVICE}", flush=True)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        t0 = time.time()
        with gpu_lock():
            loss, acc = train_one_client(self.model, self.train_loader, self.epochs)
            self.model.cpu()
            torch.cuda.empty_cache()
        print(f"[client {self.cid}] round {config.get('server_round', '?')} fit: "
              f"loss={loss:.4f} acc={acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return get_parameters(self.model), len(self.train_loader.dataset), \
            {"train_loss": loss, "train_acc": acc}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        with gpu_lock():
            val_loss, val_acc, _ = evaluate_model(self.model, self.val_loader)
            test_loss, test_acc, test_ece = evaluate_model(self.model, self.test_loader)
            self.model.cpu()
            torch.cuda.empty_cache()
        print(f"[client {self.cid}] eval (global model): val_acc={val_acc:.4f} "
              f"test_acc={test_acc:.4f} test_ece={test_ece:.4f}", flush=True)
        # loss/num_examples val-er upor; test-er number metrics-e jaay
        return val_loss, len(self.val_loader.dataset), {
            "val_acc": val_acc, "test_loss": test_loss,
            "test_acc": test_acc, "test_ece": test_ece,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True, help="client number: 1..4")
    ap.add_argument("--epochs", type=int, default=2, help="local epochs per round")
    ap.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = ap.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=FlowerClient(args.cid, args.epochs).to_client(),
    )