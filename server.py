import argparse
import csv

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays

from common import build_model, set_parameters


def weighted_avg(metrics):
    """metrics = [(num_examples, {name: value}), ...] -> example-weighted average"""
    total = sum(n for n, _ in metrics)
    names = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics) / total for k in names}


class SaveFedAvg(fl.server.strategy.FedAvg):
    """FedAvg, latest global weights mone rakhe (sheshe save korar jonno)."""
    latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = params
        return params, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--address", type=str, default="0.0.0.0:8080")
    args = ap.parse_args()

    strategy = SaveFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        on_fit_config_fn=lambda r: {"server_round": r},
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # ---- results.csv ----
    fit_m = history.metrics_distributed_fit
    eval_m = history.metrics_distributed
    rounds = [r for r, _ in eval_m["val_acc"]]
    cols = ["train_loss", "train_acc"], ["val_acc", "test_acc", "test_loss", "test_ece"]
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", *cols[0], *cols[1]])
        for i, r in enumerate(rounds):
            row = [r]
            row += [fit_m[k][i][1] if k in fit_m and i < len(fit_m[k]) else "" for k in cols[0]]
            row += [eval_m[k][i][1] for k in cols[1]]
            w.writerow(row)
    print("\nSaved results.csv")

    # ---- final global model ----
    if strategy.latest_parameters is not None:
        model = build_model()
        set_parameters(model, parameters_to_ndarrays(strategy.latest_parameters))
        torch.save(model.state_dict(), "global_model.pt")
        print("Saved global_model.pt")

    print("\nFinal round:", {k: round(v[-1][1], 4) for k, v in eval_m.items()})