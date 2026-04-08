#!/usr/bin/env python3
"""
Physical Decentralized Federated Learning — Main Entry Point

Usage (via universal launcher):
    bash run.sh

Manual usage:
    python3 main.py --node-id jetson1 \
                    --listen 0.0.0.0:50051 \
                    --self-ip <auto-detected> \
                    --peers jetson2=<peer_ip>:50051 \
                    --rounds 20 \
                    --device cuda
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from client.models.lenet import LeNetMNIST
from client.physical_client import PhysicalClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dfl.main")


# ---------------------------------------------------------------------------
# Non-IID data partitioning (Dirichlet-based, matches OCD-FL experiments)
# ---------------------------------------------------------------------------

def dirichlet_split(dataset, num_partitions: int, alpha: float = 0.5, seed: int = 42):
    """
    Split a dataset into `num_partitions` non-IID shards using a
    Dirichlet distribution over labels.  Returns a list of index arrays.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    targets = []
    for _, t in dataset:
        targets.append(int(t) if not isinstance(t, int) else t)
    targets = np.array(targets)

    num_classes = len(set(targets))
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    partitions = [[] for _ in range(num_partitions)]
    for c in range(num_classes):
        proportions = rng.dirichlet([alpha] * num_partitions)
        proportions = (proportions * len(class_indices[c])).astype(int)
        # Fix rounding
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()
        splits = np.split(
            rng.permutation(class_indices[c]),
            np.cumsum(proportions)[:-1],
        )
        for i, s in enumerate(splits):
            partitions[i].extend(s.tolist())

    return partitions


def get_partition_index(node_id: str, all_node_ids: list) -> int:
    """Deterministic mapping from node_id to partition index."""
    sorted_ids = sorted(all_node_ids)
    return sorted_ids.index(node_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Physical DFL on Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--node-id", required=True, help="Unique node identifier, e.g. jetson1")
    parser.add_argument("--listen", default="0.0.0.0:50051", help="gRPC listen address")
    parser.add_argument("--self-ip", required=True, help="This node's IP on the subnet")
    parser.add_argument(
        "--peers", nargs="*", default=[],
        help="Peer addresses as id=ip:port, e.g. jetson2=192.168.38.209:50051",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local training epochs per round")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Dirichlet concentration param (lower = more non-IID)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--selector-theta", type=float, default=0.02,
                        help="OCD-FL selector regularization")
    parser.add_argument("--sync-barrier-timeout", type=float, default=30.0,
                        help="Seconds to wait for peer readiness before each round")
    args = parser.parse_args()

    # Parse peers
    peer_addrs = {}
    all_node_ids = [args.node_id]
    for p in args.peers:
        pid, addr = p.split("=")
        peer_addrs[pid] = addr
        all_node_ids.append(pid)

    num_nodes = len(all_node_ids)
    my_index = get_partition_index(args.node_id, all_node_ids)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    partitions = dirichlet_split(full_train, num_nodes, alpha=args.dirichlet_alpha)
    my_indices = partitions[my_index]
    train_subset = Subset(full_train, my_indices)

    logger.info(
        f"[{args.node_id}] Data partition: {len(my_indices)} samples "
        f"(index={my_index}/{num_nodes})"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = LeNetMNIST(num_classes=10)
    logger.info(
        f"[{args.node_id}] LeNet params: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ------------------------------------------------------------------
    # Physical Client
    # ------------------------------------------------------------------
    client = PhysicalClient(
        node_id=args.node_id,
        listen_addr=args.listen,
        peer_addrs=peer_addrs,
        model=model,
        train_dataset=train_subset,
        test_dataset=test_set,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": args.lr, "momentum": 0.9},
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        num_classes=10,
        selector_theta=args.selector_theta,
        device=args.device,
    )

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info(f"[{args.node_id}] Caught signal {sig}, shutting down...")
        client.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client.start()

    # ------------------------------------------------------------------
    # Wait for peers to come online
    # ------------------------------------------------------------------
    logger.info(f"[{args.node_id}] Waiting for peers to come online...")
    deadline = time.time() + args.sync_barrier_timeout
    while time.time() < deadline:
        active = client.transport.discover_active_peers()
        if len(active) == len(peer_addrs):
            logger.info(f"[{args.node_id}] All peers online: {active}")
            break
        time.sleep(2)
    else:
        logger.warning(f"[{args.node_id}] Timeout waiting for peers, proceeding anyway")

    # ------------------------------------------------------------------
    # Metrics log
    # ------------------------------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    metrics_path = os.path.join(args.log_dir, f"{args.node_id}_metrics.json")
    metrics_log = []

    # ------------------------------------------------------------------
    # FL Training Loop
    # ------------------------------------------------------------------
    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"[{args.node_id}] === ROUND {round_idx}/{args.rounds} ===")

        # 1. Local training
        train_loss = client.train()

        # 2. Evaluate
        test_loss, test_acc = client.test()

        # 3. Discover neighbors (gRPC ping + metadata fetch)
        client.discover_neighbors()

        # 4. OCD-FL peer selection (knowledge gain + optimization)
        client.select_peers()

        # 5. Push model to selected peers
        client.push_model_to_peers()

        # 6. Brief wait for incoming models from peers who selected us
        logger.info(f"[{args.node_id}] Waiting for incoming models (3s)...")
        time.sleep(3)

        # 7. FedAvg aggregation over received models
        did_agg = client.aggregate()

        # 8. Post-aggregation test (optional, to measure aggregation benefit)
        if did_agg:
            post_loss, post_acc = client.test()
        else:
            post_loss, post_acc = test_loss, test_acc

        round_time = time.time() - round_start

        # Log metrics
        entry = {
            "round": round_idx,
            "train_loss": train_loss,
            "test_loss_pre_agg": test_loss,
            "test_acc_pre_agg": test_acc,
            "test_loss_post_agg": post_loss,
            "test_acc_post_agg": post_acc,
            "num_neighbors": len(client.neighbors),
            "num_peers_selected": len(client.peers),
            "aggregated": did_agg,
            "round_time_s": round_time,
        }
        metrics_log.append(entry)
        logger.info(
            f"[{args.node_id}] Round {round_idx} summary: "
            f"train_loss={train_loss:.4f}, test_acc={post_acc:.4f}, "
            f"peers={len(client.peers)}, agg={did_agg}, time={round_time:.1f}s"
        )

        # Reset per-round state
        client.neighbors = []
        client.peers = []

    # ------------------------------------------------------------------
    # Save metrics and shutdown
    # ------------------------------------------------------------------
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"[{args.node_id}] Metrics saved to {metrics_path}")

    client.stop()
    logger.info(f"[{args.node_id}] Experiment complete.")


if __name__ == "__main__":
    main()
