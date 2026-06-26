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
import concurrent.futures
import fcntl
import json
import logging
import os
import signal
import struct
import sys
import threading
import time
import socket

import matplotlib
matplotlib.use('Agg')  # headless rendering — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Subset

from client.beacon import BeaconService
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
# IID equal-class data partitioning
# ---------------------------------------------------------------------------

def iid_equal_split(targets: np.ndarray, num_partitions: int, seed: int = 42):
    """
    Split dataset indices into num_partitions IID shards with equal class
    distribution.  Each partition gets floor(class_count / num_partitions)
    samples per class.  Returns a list of index arrays (into the dataset
    whose targets were passed).
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(targets))
    partitions = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        shuffled = rng.permutation(class_idx)
        splits = np.array_split(shuffled, num_partitions)
        for i, s in enumerate(splits):
            partitions[i].extend(s.tolist())

    return partitions


def get_partition_index(node_id: str, all_node_ids: list) -> int:
    """Deterministic mapping from node_id to partition index."""
    sorted_ids = sorted(all_node_ids)
    return sorted_ids.index(node_id)


def _tcp_reachable(ip: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to ip:port succeeds within timeout."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


_SIOCGIFNETMASK = 0x891B
_SIOCGIFADDR    = 0x8915


def _get_network_prefix(my_ip: str):
    """Return (network_int, prefix_len) for the interface that owns my_ip."""
    try:
        with open("/proc/net/dev") as f:
            ifaces = [l.strip().split(":")[0].strip() for l in f if ":" in l]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for iface in ifaces:
                try:
                    ifreq = struct.pack("16sH14s", iface.encode()[:15], socket.AF_INET, b"\x00" * 14)
                    res = fcntl.ioctl(s.fileno(), _SIOCGIFADDR, ifreq)
                    if socket.inet_ntoa(res[20:24]) == my_ip:
                        res = fcntl.ioctl(s.fileno(), _SIOCGIFNETMASK, ifreq)
                        mask_int = struct.unpack("!I", res[20:24])[0]
                        ip_int = struct.unpack("!I", socket.inet_aton(my_ip))[0]
                        return ip_int & mask_int, bin(mask_int).count("1")
                except OSError:
                    continue
    except Exception:
        pass
    # Fall back to /24
    net = ".".join(my_ip.split(".")[:3]) + ".0"
    return struct.unpack("!I", socket.inet_aton(net))[0], 24


def scan_for_peers(my_ip: str, port: int, timeout: float = 0.1) -> dict:
    """
    Parallel scan of the full local network for active gRPC peers.
    Automatically detects the subnet (e.g. /18) so Jetsons on different
    /24s within the same network are discovered without any manual config.
    Uses conservative concurrency (64 workers) to avoid flooding the AP.
    Returns {peer_id: "ip:port"} for hosts that accept a TCP connection.
    """
    net_int, prefix_len = _get_network_prefix(my_ip)
    num_hosts = (1 << (32 - prefix_len)) - 2
    ips = [
        socket.inet_ntoa(struct.pack("!I", net_int + i))
        for i in range(1, num_hosts + 1)
        if socket.inet_ntoa(struct.pack("!I", net_int + i)) != my_ip
    ]

    candidates = {}
    lock = threading.Lock()

    def _check(ip):
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                peer_id = "jetson_" + ip.replace(".", "_")
                with lock:
                    candidates[peer_id] = f"{ip}:{port}"
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        ex.map(_check, ips)

    return candidates


def _verify_dfl_peer(ip: str, port: int, timeout: float = 1.0) -> bool:
    """
    Return True only if the host at ip:port responds to our gRPC DFL Ping.
    Filters out unrelated school-network services that happen to use port 50051.
    """
    try:
        import grpc
        from client.transport import dfl_pb2, dfl_pb2_grpc
        with grpc.insecure_channel(f"{ip}:{port}") as ch:
            stub = dfl_pb2_grpc.PeerServiceStub(ch)
            resp = stub.Ping(
                dfl_pb2.PingRequest(sender_id="scanner", sender_addr="0.0.0.0:0"),
                timeout=timeout,
            )
            return resp.peer_id != ""
    except Exception:
        return False


def _background_scanner(my_ip: str, port: int, transport, stop_event: threading.Event,
                        interval: float = 30.0):
    """
    Daemon thread: periodically rescans the network and adds newly discovered
    DFL peers to the live transport. Handles Jetsons that join after startup.
    """
    while not stop_event.wait(interval):
        candidates = scan_for_peers(my_ip, port)
        for pid, addr in candidates.items():
            ip = addr.split(":")[0]
            if _verify_dfl_peer(ip, port):
                transport.add_peer(pid, addr)


# ---------------------------------------------------------------------------
# Post-training plots (headless — saved as PNG, no display needed)
# ---------------------------------------------------------------------------

def save_plots(metrics_log: list, node_id: str, log_dir: str):
    """Generate and save a 2×3 metrics summary PNG after training completes."""
    rounds = [e["round"] for e in metrics_log]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"DFL Training — {node_id}", fontsize=14)

    # Train loss (final epoch per round)
    axes[0, 0].plot(rounds, [e["train_loss"] for e in metrics_log], marker="o")
    axes[0, 0].set_title("Train Loss (final epoch)")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Loss")

    # Per-epoch losses flattened across all rounds
    ax = axes[0, 1]
    for e in metrics_log:
        epoch_losses = e.get("epoch_losses", [e["train_loss"]])
        start = (e["round"] - 1) * len(epoch_losses)
        xs = range(start, start + len(epoch_losses))
        ax.plot(xs, epoch_losses, color="steelblue", alpha=0.6)
    ax.set_title("Per-Epoch Train Loss")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Loss")

    # Test accuracy pre/post aggregation
    axes[0, 2].plot(rounds, [e["test_acc_pre_agg"]  for e in metrics_log], marker="o", label="Pre-agg")
    axes[0, 2].plot(rounds, [e["test_acc_post_agg"] for e in metrics_log], marker="s", label="Post-agg")
    axes[0, 2].set_title("Test Accuracy")
    axes[0, 2].set_xlabel("Round")
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].legend()

    # Test loss pre/post aggregation
    axes[1, 0].plot(rounds, [e["test_loss_pre_agg"]  for e in metrics_log], marker="o", label="Pre-agg")
    axes[1, 0].plot(rounds, [e["test_loss_post_agg"] for e in metrics_log], marker="s", label="Post-agg")
    axes[1, 0].set_title("Test Loss")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Learning rate
    axes[1, 1].plot(rounds, [e["learning_rate"] for e in metrics_log], marker="o", color="orange")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("LR")

    # Peers selected (bar) + round time (line, secondary axis)
    ax2 = axes[1, 2].twinx()
    axes[1, 2].bar(rounds, [e["num_peers_selected"] for e in metrics_log], alpha=0.6, label="Peers selected")
    ax2.plot(rounds, [e["round_time_s"] for e in metrics_log], color="red", marker="o", label="Round time (s)")
    axes[1, 2].set_title("Peers Selected & Round Time")
    axes[1, 2].set_xlabel("Round")
    axes[1, 2].set_ylabel("Peers")
    ax2.set_ylabel("Time (s)")
    axes[1, 2].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(log_dir, f"{node_id}_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{node_id}] Plots saved to {out_path}")


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--selector-theta", type=float, default=0.02,
                        help="OCD-FL selector regularization")
    parser.add_argument("--selector-gamma", type=float, default=0.3,
                        help="Weight for EMD-based data diversity in peer selection")
    parser.add_argument("--sync-barrier-timeout", type=float, default=30.0,
                        help="Seconds to wait for peer readiness before each round")
    parser.add_argument("--dataset", default="FashionMNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"],
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

    # Derive partition index from sorted position of self_ip among all known IPs.
    # Sorting full IPs (not just last octet) gives unique, stable assignments
    # even when all nodes share the same last octet.
    peer_ips = [addr.split(":")[0] for addr in peer_addrs.values()]
    all_ips_sorted = sorted([args.self_ip] + peer_ips)
    my_index = all_ips_sorted.index(args.self_ip) % num_nodes

    # ------------------------------------------------------------------
    # Data — FashionMNIST 10k sample with IID equal-class split
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # Load full FashionMNIST: 60k train + 10k test = 70k total
    train_raw = datasets.FashionMNIST(root=args.data_dir, train=True,  download=True, transform=transform)
    test_raw  = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_raw, test_raw])   # 70 000 samples

    # Extract all 70k targets without iterating the dataset (fast)
    all_targets = np.concatenate([
        train_raw.targets.numpy(),   # (60000,)
        test_raw.targets.numpy(),    # (10000,)
    ])

    # Sample 10k reproducibly (same indices on every node)
    SAMPLE_SIZE = 10_000
    rng_sample = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(len(full_dataset), generator=rng_sample)[:SAMPLE_SIZE].numpy()
    sampled_targets = all_targets[sample_indices]

    # 90/10 train/test split of the 10k sample
    TRAIN_SIZE = int(0.9 * SAMPLE_SIZE)   # 9000
    TEST_SIZE  = SAMPLE_SIZE - TRAIN_SIZE  # 1000

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(SAMPLE_SIZE)
    train_pool_local_idx = perm[:TRAIN_SIZE]   # indices into sample_indices
    test_pool_local_idx  = perm[TRAIN_SIZE:]

    train_pool_global = sample_indices[train_pool_local_idx]  # indices into full_dataset
    test_pool_global  = sample_indices[test_pool_local_idx]

    train_pool = Subset(full_dataset, train_pool_global.tolist())
    test_set   = Subset(full_dataset, test_pool_global.tolist())  # 1000 samples, shared

    train_targets_pool = sampled_targets[train_pool_local_idx]   # (9000,)

    # IID equal-class partition of the 9k training pool across all nodes
    partitions = iid_equal_split(train_targets_pool, num_nodes)
    my_local_indices = partitions[my_index]        # indices into train_pool
    train_subset = Subset(train_pool, my_local_indices)

    logger.info(
        f"[{args.node_id}] Data: FashionMNIST 10k sample → "
        f"{TRAIN_SIZE} train / {TEST_SIZE} test | "
        f"partition {my_index}/{num_nodes}: {len(my_local_indices)} samples"
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

    beacon = None  # assigned after peer discovery; beacon threads are daemons so they die either way
    _scanner_stop = threading.Event()

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info(f"[{args.node_id}] Caught signal {sig}, shutting down...")
        _scanner_stop.set()
        if beacon is not None:
            beacon.stop()
        client.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client.start()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=5, gamma=0.5)

    # ------------------------------------------------------------------
    # Wait for peers to come online
    # ------------------------------------------------------------------
    PORT = int(args.listen.split(":")[-1])
    expected_peers = (args.total_nodes - 1) if args.total_nodes else 1

    if peer_addrs:
        # Explicit peers provided via --peers / PEER_IPS — wait for them to
        # come online but skip the network scan entirely.
        logger.info(
            f"[{args.node_id}] Explicit peer list provided: {list(peer_addrs.keys())}. "
            f"Skipping subnet scan."
        )
        deadline = time.time() + args.sync_barrier_timeout
        while time.time() < deadline:
            reachable = {
                pid: addr for pid, addr in peer_addrs.items()
                if _tcp_reachable(addr.split(":")[0], PORT)
            }
            logger.info(
                f"[{args.node_id}] Reachable peers: {list(reachable.keys())} "
                f"({len(reachable)}/{expected_peers} expected)"
            )
            if len(reachable) >= expected_peers:
                break
            logger.info(f"[{args.node_id}] Waiting for peers... retrying in 5s")
            time.sleep(5)
        # Use the full explicit list (not just reachable) — unreachable peers
        # will simply fail at Ping time each round and be skipped gracefully.
        final_peers = peer_addrs
    else:
        # No explicit peers — scan the full local network (auto-detects /18, /24, etc.)
        logger.info(f"[{args.node_id}] Scanning local network for peers...")
        deadline = time.time() + args.sync_barrier_timeout
        final_peers = {}
        while time.time() < deadline:
            candidates = scan_for_peers(args.self_ip, PORT)
            # Filter out non-DFL services that happen to have port 50051 open
            final_peers = {
                pid: addr for pid, addr in candidates.items()
                if _verify_dfl_peer(addr.split(":")[0], PORT)
            }
            if len(candidates) != len(final_peers):
                logger.info(
                    f"[{args.node_id}] Filtered {len(candidates) - len(final_peers)} "
                    f"non-DFL host(s) from scan results"
                )
            logger.info(
                f"[{args.node_id}] Found {len(final_peers)} peer(s): {list(final_peers.keys())}"
            )
            if len(final_peers) >= expected_peers:
                break
            logger.info(f"[{args.node_id}] Waiting for peers... retrying in 5s")
            time.sleep(5)

    client.transport.update_peers(final_peers)

    # ------------------------------------------------------------------
    # Beacon — zero-config peer discovery (runs for the lifetime of training)
    # ------------------------------------------------------------------
    beacon = BeaconService(
        node_id=args.node_id,
        my_ip=args.self_ip,
        grpc_port=PORT,
        on_peer_discovered=client.transport.add_peer,
    )
    beacon.start()

    # Background scanner: re-scans every 30s to find Jetsons that join after startup
    threading.Thread(
        target=_background_scanner,
        args=(args.self_ip, PORT, client.transport, _scanner_stop),
        daemon=True,
        name="bg-scanner",
    ).start()

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
        train_loss, epoch_losses = client.train()

        # 3. Evaluate
        test_loss, test_acc = client.test()

        # 4. OCD-FL peer selection (knowledge gain + optimization)
        client.select_peers()

        # 5. Push model to all discovered neighbors
        client.push_model_to_all_neighbors()

        # 6. Brief wait for incoming models from peers who selected us
        logger.info(f"[{args.node_id}] Waiting for incoming models (3s)...")
        time.sleep(3)

        # 7. FedAvg aggregation over received models
        did_agg = client.aggregate()
        lr_scheduler.step()  # Step LR scheduler after aggregation (even if no agg happened)

        # 8. Post-aggregation test
        if did_agg:
            post_loss, post_acc = client.test()
        else:
            post_loss, post_acc = test_loss, test_acc

        round_time = time.time() - round_start

        # Log metrics
        current_lr = client.optimizer.param_groups[0]["lr"]
        entry = {
            "round": round_idx,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "epoch_losses": epoch_losses,
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
            f"lr={current_lr}, train_loss={train_loss:.4f}, test_acc={post_acc:.4f}, "
            f"peers={len(client.peers)}, agg={did_agg}, time={round_time:.1f}s"
        )

        # Reset per-round state
        client.neighbors = []
        client.peers = []

    # ------------------------------------------------------------------
    # Save metrics and plots, then shutdown
    # ------------------------------------------------------------------
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"[{args.node_id}] Metrics saved to {metrics_path}")

    save_plots(metrics_log, args.node_id, args.log_dir)

    client.stop()
    logger.info(f"[{args.node_id}] Experiment complete.")


if __name__ == "__main__":
    main()
