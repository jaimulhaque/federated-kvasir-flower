# Federated Learning on Kvasir v2 with Flower

A small federated learning (FL) test using the [Flower](https://flower.ai) framework.
EfficientNet-B0 is trained on the Kvasir v2 gastrointestinal endoscopy dataset (8 classes, 8000 images)
with **4 simulated clients running on one machine**, aggregated with FedAvg.

## Setup

- Kvasir v2 is split into 4 clients of 2000 images each (250 per class, stratified).
- Each client splits its own data 70/15/15 into train/val/test (1400 / 296 / 304 images).
- Model: EfficientNet-B0 (ImageNet pretrained), 224x224 input, AdamW (lr 1e-4), mixed precision.
- Each round the global model is evaluated on every client's own val and test sets:
  accuracy, loss and Expected Calibration Error (ECE).
- Clients take turns on the GPU through a file lock, so 4 clients fit on a single 8 GB GPU.

## Files

| File | Purpose |
|---|---|
| `split_data.py` | Splits the raw dataset into `clients/client_1..4/{train,val,test}` |
| `common.py` | Data loaders, model, training, evaluation (accuracy, ECE) |
| `client.py` | Flower client (`--cid 1..4`) |
| `server.py` | Flower server, FedAvg, writes `results.csv` and `global_model.pt` |
| `run_all.sh` | Starts the server and 4 clients |

## Usage

```bash
pip install torch torchvision flwr==1.13.1

# 1. put Kvasir v2 in RAW DATASET/kvasir-dataset-v2 and edit paths in split_data.py, then:
python split_data.py

# 2. quick smoke test (2 rounds, 1 local epoch), then a full run (10 rounds, 2 local epochs)
bash run_all.sh 2 1
bash run_all.sh 10 2
```

Set `FL_DATA_ROOT` to point at a different `clients` folder.
Logs go to `logs/`, per-round metrics to `results.csv`.

## Data

The dataset is not included. Download Kvasir v2 from the official source:
https://datasets.simula.no/kvasir/

## Notes

Tested with Python 3.12, PyTorch 2.13 (CUDA 12.6), Flower 1.13.1 on WSL2 (Ubuntu), RTX 3050 8 GB.
