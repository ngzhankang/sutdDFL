"""
PhysicalClient: A federated learning client for real Jetson-to-Jetson deployment.

Key differences from the simulator's Client:
  - No in-memory list of clients; neighbors are discovered via gRPC Ping.
  - knowledge_gain() fetches remote metadata over the network.
  - Aggregation consumes models received asynchronously from the gRPC buffer.
  - Label distribution is computed locally for EMD-based Data Distribution Difference.
  - The EfficientPeerSelector from OCD-FL is preserved and operates on RemotePeerProxy objects.
"""

import logging
import math
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from client.transport.grpc_transport import GrpcTransport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Remote peer proxy — lightweight stand-in so EfficientPeerSelector works
# ---------------------------------------------------------------------------

class RemotePeerProxy:
    """
    Mimics the interface that EfficientPeerSelector.select() and
    Client.knowledge_gain() expect from a neighbor, but backed by
    remote metadata fetched over gRPC.
    """

    def __init__(self, peer_id: str, meta: dict):
        self.id_ = peer_id
        self.peer_id = peer_id
        # Reconstruct the fields the selector reads
        self.loss_current = meta["loss_current"]
        self.loss_prev = meta["loss_prev"]
        self.label_dist = np.array(meta["label_dist"], dtype=np.float64)
        self.cpu_frequency = meta["cpu_frequency"]
        self.idle_time = meta["idle_time"]

        # Fake model attribute with loss_history for knowledge_gain()
        class _FakeLossModel:
            def __init__(self, prev, cur):
                self.loss_history = (prev, cur)
        self.model = _FakeLossModel(meta["loss_prev"], meta["loss_current"])

    def __str__(self):
        return f"RemotePeer({self.peer_id})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, RemotePeerProxy):
            return self.peer_id == other.peer_id
        return False

    def __hash__(self):
        return hash(self.peer_id)


# ---------------------------------------------------------------------------
# EMD (Earth Mover Distance) for Data Distribution Difference
# ---------------------------------------------------------------------------

def compute_label_distribution(dataset: Dataset, num_classes: int = 10) -> np.ndarray:
    """Compute normalized label frequency vector from a dataset/subset."""
    counts = Counter()
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        counts[int(label)] += 1
    total = sum(counts.values())
    dist = np.zeros(num_classes, dtype=np.float64)
    for cls, cnt in counts.items():
        dist[cls] = cnt / total
    return dist


def earth_mover_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    1D Earth Mover Distance (Wasserstein-1) between two discrete distributions.
    This is the metric L(D_Vk, D_G) from the midterm slides:
        L = sum_j | p_j - q_j |
    """
    return float(np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# PhysicalClient
# ---------------------------------------------------------------------------

class PhysicalClient:
    """
    Federated learning client for physical Jetson deployment.

    Parameters
    ----------
    node_id : str
        Unique identifier, e.g. "jetson1".
    listen_addr : str
        gRPC listen address, e.g. "0.0.0.0:50051".
    peer_addrs : dict
        Mapping of peer_id -> "ip:port", e.g. {"jetson2": "192.168.38.209:50051"}.
    model : torch.nn.Module
        The local neural network (LeNet for MNIST).
    train_dataset : Dataset
        Local training data shard.
    test_dataset : Dataset
        Global test set for evaluation.
    optimizer_cls : type
        Optimizer class (default: torch.optim.SGD).
    optimizer_kwargs : dict
        Kwargs for the optimizer.
    local_epochs : int
        Number of local training epochs per round.
    batch_size : int
        DataLoader batch size.
    num_classes : int
        Number of classification labels (10 for MNIST).
    selector_theta : float
        Regularization parameter for EfficientPeerSelector.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        node_id: str,
        listen_addr: str,
        peer_addrs: Dict[str, str],
        model: torch.nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
        local_epochs: int = 3,
        batch_size: int = 64,
        num_classes: int = 10,
        selector_theta: float = 0.02,
        selector_gamma: float = 0.3,
        device: str = "cpu",
    ):
        self.node_id = node_id
        self.device = torch.device(device)

        # Model
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        opt_kw = optimizer_kwargs or {"lr": 0.01, "momentum": 0.9}
        self.optimizer = optimizer_cls(self.model.parameters(), **opt_kw)

        # Data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.num_classes = num_classes

        # Label distribution for EMD
        self.label_dist = compute_label_distribution(train_dataset, num_classes)

        # Loss tracking (mirrors simulator's loss_history tuple)
        self.loss_history: Tuple[float, float] = (float("inf"), float("inf"))

        # OCD-FL peer selection
        self.selector_theta = selector_theta
        self.selector_gamma = selector_gamma
        self.neighbors: List[RemotePeerProxy] = []
        self.peers: List[RemotePeerProxy] = []
        self.is_active = True

        # Timing for idle-time reporting
        self._last_train_end = time.time()

        # CPU frequency (read from Jetson sysfs if available, fallback)
        self._cpu_freq = self._read_cpu_freq()

        # Aggregation buffer: models received asynchronously
        self._received_models: List[Tuple[str, dict, float]] = []

        # Transport
        self.transport = GrpcTransport(
            node_id=node_id,
            listen_addr=listen_addr,
            peer_addrs=peer_addrs,
            get_meta_cb=self._get_meta,
            on_model_received_cb=self._on_model_received,
            is_active_cb=lambda: self.is_active,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the gRPC server (call once at boot)."""
        self.transport.start()
        logger.info(f"[{self.node_id}] PhysicalClient started")

    def stop(self):
        """Gracefully stop the gRPC server."""
        self.transport.stop()
        logger.info(f"[{self.node_id}] PhysicalClient stopped")

    # ------------------------------------------------------------------
    # Callbacks for GrpcTransport
    # ------------------------------------------------------------------

    def _get_meta(self) -> dict:
        """Called by gRPC servicer when a peer requests our metadata."""
        return {
            "loss_prev": float(self.loss_history[0]),
            "loss_current": float(self.loss_history[1]),
            "label_dist": self.label_dist.tolist(),
            "cpu_frequency": self._cpu_freq,
            "idle_time": time.time() - self._last_train_end,
        }

    def _on_model_received(self, sender_id: str, state_dict: dict, loss: float):
        """Called by gRPC servicer when a peer pushes a model to us."""
        self._received_models.append((sender_id, state_dict, loss))
        logger.info(f"[{self.node_id}] Buffered model from {sender_id}")

    # ------------------------------------------------------------------
    # Peer Discovery (replaces simulator's lookup + relocate)
    # ------------------------------------------------------------------

    def discover_neighbors(self) -> List[RemotePeerProxy]:
        """
        Ping all known peers over gRPC, fetch metadata from active ones,
        and build RemotePeerProxy objects that the selector can operate on.
        """
        self.neighbors = []
        active_peer_ids = self.transport.discover_active_peers()
        logger.info(f"[{self.node_id}] Active peers: {active_peer_ids}")

        for pid in active_peer_ids:
            meta = self.transport.fetch_meta(pid)
            if meta is not None:
                proxy = RemotePeerProxy(pid, meta)
                self.neighbors.append(proxy)

        logger.info(
            f"[{self.node_id}] Discovered {len(self.neighbors)} neighbor(s): "
            f"{[n.peer_id for n in self.neighbors]}"
        )
        return self.neighbors

    # ------------------------------------------------------------------
    # OCD-FL Knowledge Gain (preserved from original codebase)
    # ------------------------------------------------------------------

    def knowledge_gain(self, neighbor: RemotePeerProxy) -> float:
        """
        Compute knowledge gain with a remote neighbor.
        Identical logic to the simulator:
            KG = (1 - exp(-2 * delta)) * (delta > 0)
        where delta = neighbor_loss - self_loss.
        """
        ln = self.loss_history[-1]
        lk = neighbor.model.loss_history[-1]
        delta = lk - ln
        return float((1 - math.exp(-2 * delta)) * (delta > 0))

    def data_distribution_difference(self, neighbor: RemotePeerProxy) -> float:
        """
        EMD-based Data Distribution Difference from the midterm slides:
            L(D_Vk, D_G) = sum_j | l_j^Vk / |D_Vk| - l_j^G / |D_G| |
        Here we compare our local distribution with the neighbor's.
        """
        return earth_mover_distance(self.label_dist, neighbor.label_dist)

    # ------------------------------------------------------------------
    # Communication energy proxy (simplified for physical deployment)
    # ------------------------------------------------------------------

    def communication_cost(self, neighbor: RemotePeerProxy) -> float:
        """
        Proxy for communication energy. In the physical deployment we use
        model size / estimated bandwidth as a cost metric, plus a penalty
        proportional to the neighbor's idle time (straggler avoidance).
        """
        model_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)
        # Rough estimate: 5 MB/s over mobile hotspot WiFi
        estimated_bw_mbps = 5.0
        transfer_time = model_size_mb / estimated_bw_mbps
        # Penalize stragglers
        idle_penalty = max(0.0, neighbor.idle_time - 10.0) * 0.1
        return transfer_time + idle_penalty

    # ------------------------------------------------------------------
    # OCD-FL Peer Selection (EfficientPeerSelector logic preserved)
    # ------------------------------------------------------------------

    def select_peers(self) -> List[RemotePeerProxy]:
        """
        Run the OCD-FL optimization-based peer selection.
        This is a self-contained reimplementation of EfficientPeerSelector
        that works with RemotePeerProxy objects and the physical cost model.
        """
        if not self.neighbors:
            self.peers = []
            return self.peers

        # Compute knowledge gain and cost for each neighbor
        kgains = [self.knowledge_gain(n) for n in self.neighbors]
        costs = [self.communication_cost(n) for n in self.neighbors]

        # After computing kgains and costs, add EMD as a bonus signal
        emds = [self.data_distribution_difference(n) for n in self.neighbors]
        max_emd = max(emds) if max(emds) > 0 else 1.0
        emd_norm = [e / max_emd for e in emds]

        # Normalize costs to [0, 1]
        max_cost = max(costs) if max(costs) > 0 else 1.0
        costs_norm = [c / max_cost for c in costs]

        t_kgain = torch.tensor(kgains, dtype=torch.float32)
        t_cost = torch.tensor(costs_norm, dtype=torch.float32)

        # EMD: prefer peers with different data distributions
        emds = [self.data_distribution_difference(n) for n in self.neighbors]
        max_emd = max(emds) if max(emds) > 0 else 1.0
        emd_norm = [e / max_emd for e in emds]
        t_emd = torch.tensor(emd_norm, dtype=torch.float32)


        # Optimization (from EfficientPeerSelector)
        n = len(self.neighbors)
        weights = torch.nn.Parameter(torch.rand(n))
        optimizer = torch.optim.Adam([weights], lr=0.1, maximize=True)

        theta = self.selector_theta
        best_loss = float("-inf")
        patience_counter = 0

        for epoch in range(500):
            optimizer.zero_grad()
            betas = torch.sigmoid(weights)
            reward = torch.dot(betas, t_kgain + self.selector_gamma * t_emd) / (torch.dot(betas, t_cost) + 1e-8)
            reg = theta * torch.norm(weights, p=2)
            loss = reward + reg
            loss.backward()
            optimizer.step()

            # Early stopping
            current = loss.item()
            if abs(current - best_loss) < 1e-4:
                patience_counter += 1
                if patience_counter >= 5:
                    break
            else:
                patience_counter = 0
            best_loss = current

        mask = torch.sigmoid(weights).detach().numpy().round().astype(bool)
        self.peers = [self.neighbors[i] for i in range(n) if mask[i]]

        logger.info(
            f"[{self.node_id}] Selected {len(self.peers)} peer(s): "
            f"{[p.peer_id for p in self.peers]}"
        )
        return self.peers

    # ------------------------------------------------------------------
    # Local Training (LeNet on MNIST, edge-train-bench style)
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        Execute local training for `local_epochs` epochs.
        Returns the average training loss of the last epoch.
        """
        self.model.train()
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )

        epoch_loss = 0.0
        for epoch in range(self.local_epochs):
            running_loss = 0.0
            n_batches = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            logger.info(
                f"[{self.node_id}] Epoch [{epoch+1}/{self.local_epochs}] "
                f"train_loss={epoch_loss:.4f}"
            )

        # Update loss history
        self.loss_history = (self.loss_history[1], epoch_loss)
        self._last_train_end = time.time()
        return epoch_loss

    def test(self) -> Tuple[float, float]:
        """Evaluate on the test set. Returns (loss, accuracy)."""
        self.model.eval()
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.loss_fn(outputs, targets).item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        # Update loss history from test (matches simulator behavior)
        self.loss_history = (self.loss_history[1], avg_loss)
        logger.info(
            f"[{self.node_id}] Test loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
        )
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Aggregation (FedAvg over physically received models)
    # ------------------------------------------------------------------

    def push_model_to_peers(self):
        """Push our current model to all selected peers."""
        sd = self.model.state_dict()
        loss = self.loss_history[-1]
        for peer in self.peers:
            self.transport.push_model(peer.peer_id, sd, loss)

    def aggregate(self) -> bool:
        """
        FedAvg aggregation over locally-buffered models received from peers.
        Returns True if aggregation happened.
        """
        # Drain the gRPC receive buffer
        received = self.transport.drain_received_models()
        received += list(self._received_models)
        self._received_models.clear()

        if not received:
            logger.info(f"[{self.node_id}] No models to aggregate")
            return False

        # FedAvg: average own model with received models
        own_sd = self.model.state_dict()
        all_sds = [own_sd] + [sd for _, sd, _ in received]

        agg_sd = {}
        for key in own_sd:
            stacked = torch.stack(
                [sd[key].float().to(self.device) for sd in all_sds if key in sd]
            )
            agg_sd[key] = stacked.mean(dim=0)

        self.model.load_state_dict(agg_sd)
        logger.info(
            f"[{self.node_id}] Aggregated {len(received)} received model(s) "
            f"with local model (FedAvg)"
        )
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _read_cpu_freq() -> float:
        """Try to read the Jetson's current CPU frequency from sysfs."""
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                return float(f.read().strip()) * 1000  # kHz -> Hz
        except Exception:
            return 1.5e9  # fallback: 1.5 GHz
