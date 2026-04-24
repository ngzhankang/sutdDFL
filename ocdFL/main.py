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
import socket

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


def scan_for_peers(subnet: str, port: int, my_ip: str, timeout: float = 0.3) -> dict:
    """Scan subnet for active gRPC peers. Returns {peer_id: ip:port}."""
    peers = {}
    for i in range(1, 255):
        ip = f"{subnet}.{i}"
        if ip == my_ip:
            continue
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                peer_id = f"jetson_{ip.split('.')[-1]}"
                peers[peer_id] = f"{ip}:{port}"
        except Exception:
            pass
    return peers

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
    parser.add_argument("--selector-gamma", type=float, default=0.3,
                        help="Weight for EMD-based data diversity in peer selection")
    parser.add_argument("--sync-barrier-timeout", type=float, default=30.0,
                        help="Seconds to wait for peer readiness before each round")
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"],
                        help="Dataset to use for training")
    parser.add_argument("--total-nodes", type=int, default=None,
                    help="Total nodes in cluster (for data partitioning)")
    args = parser.parse_args()

    # Parse peers
    peer_addrs = {}
    all_node_ids = [args.node_id]
    for p in args.peers:
        pid, addr = p.split("=")
        peer_addrs[pid] = addr
        all_node_ids.append(pid)

    num_nodes = args.total_nodes if args.total_nodes else len(all_node_ids)
    
    def ip_based_index(self_ip: str, total_nodes: int) -> int:
        """Use last octet mod total_nodes for stable partition assignment."""
        last_octet = int(self_ip.split(".")[-1])
        return last_octet % total_nodes

    my_index = ip_based_index(args.self_ip, num_nodes)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    DatasetClass = getattr(datasets, args.dataset)

    if args.dataset == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    full_dataset = DatasetClass(root=args.data_dir, train=True, download=True, transform=transform)

    # 90/10 train/test split
    total = len(full_dataset)
    train_size = int(0.9 * total)
    test_size = total - train_size
    full_train, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

     # Build label list for partitioning
    class LabeledSubset:
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            return self.subset[idx]

    partitions = dirichlet_split(full_dataset, num_nodes, alpha=args.dirichlet_alpha)
    my_indices = partitions[my_index]
    # Remap indices to the 70% train subset
    train_indices_set = set(full_train.indices)
    my_train_indices = [i for i in my_indices if i in train_indices_set]
    train_subset = Subset(full_dataset, my_train_indices)

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
        selector_gamma=args.selector_gamma,
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
    # Scan for peers now that our gRPC server is up
    subnet = ".".join(args.self_ip.split(".")[:3])
    logger.info(f"[{args.node_id}] Scanning for peers on {subnet}.0/24...")
    
    deadline = time.time() + args.sync_barrier_timeout

    # Initial peer_addrs from --peers args (empty when launched via run.sh)
    # Will be overwritten by subnet scan below
    PORT = int(args.listen.split(":")[-1])

    while time.time() < deadline:
        peer_addrs = scan_for_peers(subnet, PORT, args.self_ip)
        logger.info(f"[{args.node_id}] Found {len(peer_addrs)} peer(s): {list(peer_addrs.keys())}")
        expected_peers = (args.total_nodes - 1) if args.total_nodes else 1
        if len(peer_addrs) >= expected_peers:
            break
        logger.info(f"[{args.node_id}] Waiting for peers... retrying in 5s")
        time.sleep(5)
    
    # Update transport with discovered peers
    client.transport.update_peers(peer_addrs)

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

        # 1. Discover neighbors (gRPC ping + metadata fetch)
        client.discover_neighbors()
        
        # 2. Local training
        train_loss = client.train()

        # 3. Evaluate
        test_loss, test_acc = client.test()

        # 4. OCD-FL peer selection (knowledge gain + optimization)
        client.select_peers()

        # 5. Push model to selected peers
        # client.push_model_to_peers()
        client.push_model_to_all_neighbors()

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
