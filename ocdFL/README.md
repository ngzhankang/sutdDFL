# Physical Decentralized Federated Learning on Jetson Orin Nano

This codebase converts the [nizar-masmoudi/decentralized-federated-learning](https://github.com/nizar-masmoudi/decentralized-federated-learning) simulator into a **real physical P2P deployment** across Nvidia Jetson Orin Nanos.

## Architecture Overview

```
┌─────────────────────────────────┐        gRPC over WiFi        ┌─────────────────────────────────┐
│  Jetson A  (auto-detected IP)   │◄────────────────────────────►│  Jetson B  (auto-detected IP)   │
│                                 │   Ping / ExchangeMeta /      │                                 │
│  PhysicalClient("sutdjetson1")  │   PushModel                  │  PhysicalClient("sutdjetson2")  │
│  ├─ LeNet-5 (MNIST)             │                              │  ├─ LeNet-5 (MNIST)             │
│  ├─ gRPC server :50051          │                              │  ├─ gRPC server :50051          │
│  ├─ OCD-FL peer selector        │                              │  ├─ OCD-FL peer selector        │
│  └─ FedAvg aggregator           │                              │  └─ FedAvg aggregator           │
└─────────────────────────────────┘                              └─────────────────────────────────┘
```

Scales to N Jetsons — each node auto-discovers all peers on the same subnet.

## What Changed vs. the Simulator

| Aspect | Simulator | Physical Deployment |
|--------|-----------|-------------------|
| Neighbors | `client.lookup(clients)` iterates in-memory list | `discover_neighbors()` pings peers via gRPC |
| Model exchange | `sender.model` read from Python reference | `PushModel` RPC serializes state_dict to protobuf |
| Knowledge gain | `neighbor.model.loss_history` accessed directly | `ExchangeMeta` RPC fetches loss/label_dist remotely |
| Data Distribution Difference | Not explicitly used | EMD integrated into peer selection optimizer via `gamma` weight |
| Aggregation | Synchronous loop over `active_clients` | Asynchronous: models buffered, FedAvg after timeout |
| Peer selection | `EfficientPeerSelector` on `Client` objects | Same optimization, operates on `RemotePeerProxy` |

## File Structure

```
ocdFL/
├── main.py                      # Entry point (run on each Jetson)
├── run.sh                       # Universal launcher (auto-detects IP, hostname, peers)
├── compile_protos.sh            # Compiles .proto → Python gRPC stubs
├── requirements.txt
├── protos/
│   └── dfl.proto                # gRPC service definition
└── client/
    ├── __init__.py
    ├── physical_client.py       # Core: PhysicalClient + RemotePeerProxy + EMD
    ├── models/
    │   ├── __init__.py
    │   └── lenet.py             # LeNet-5 for MNIST (edge-train-bench style)
    └── transport/
        ├── __init__.py
        └── grpc_transport.py    # gRPC server/client transport layer
```

## Prerequisites
- At least one Jetson running a Docker container with `--net=host --dns 8.8.8.8` (e.g. `docker run --net=host --dns 8.8.8.8 -v /home/$USER/SUTD:/app <your_image>`)
- All Jetsons on the same WiFi network (hotspot or router — IPs are auto-detected)
- Working directory `/app` mapped to host's `/home/$USER/SUTD`

## Quick Start

### 1. Install dependencies (inside Docker container)

```bash
pip install -r requirements.txt
```

If you hit DNS issues on a mobile hotspot:
```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
pip install -r requirements.txt
```

### 2. Launch on every Jetson

The same command on every node:

```bash
bash run.sh
```

`run.sh` automatically:
- Derives the node ID from hostname (e.g. `sutdJetson1` → `sutdjetson1`)
- Detects the local IP via `hostname -I`
- Scans the subnet for other Jetsons listening on port 50051
- Passes everything to `main.py`

**Start the first Jetson, then the rest.** The first node listens immediately; subsequent nodes discover it on boot.

### 3. Check results

Metrics are saved to `./logs/<node_id>_metrics.json` on each Jetson, e.g.:

```bash
cat logs/sutdjetson1_metrics.json | python3 -m json.tool
```

Each entry contains per-round train loss, test accuracy (pre/post aggregation), number of peers selected, and round time.

## Dataset Selection
 
Supported datasets: `MNIST`, `FashionMNIST`, `CIFAR10`. Datasets download automatically on first run.
 
```bash
DATASET=FashionMNIST bash run.sh
```
 
Or manually:
 
```bash
python3 main.py --node-id sutdjetson1 --self-ip 192.168.1.86 --dataset FashionMNIST ...
```
 
The data pipeline uses a **70/30 train/test split** applied before partitioning across nodes.
 
## Data Partitioning
 
Training data is split across all discovered nodes using a **Dirichlet distribution**. The `--dirichlet-alpha` (or `ALPHA` env var) controls how the data is distributed:
 
| Alpha Value | Distribution | Use Case |
|-------------|-------------|----------|
| `0.1` | Highly non-IID (each node gets mostly 1–2 classes) | Stress-test federated aggregation |
| `0.5` | Moderately non-IID (default) | Realistic edge scenario |
| `1.0` | Mildly non-IID | Moderate heterogeneity |
| `10000` | Near-uniform IID (equal split) | Baseline comparison |
 
**Example: 3 Jetsons with equal IID split on FashionMNIST:**
 
```bash
DATASET=FashionMNIST ALPHA=10000 bash run.sh
```
 
Each node deterministically receives its partition based on sorted hostname, so all nodes get complementary (non-overlapping) data shards without coordination.
 
## Configuration Reference
 
All parameters can be set via environment variables in `run.sh` or passed directly to `main.py`.
 
### Environment Variables (for `run.sh`)
 
```bash
DATASET=FashionMNIST ROUNDS=30 LOCAL_EPOCHS=5 BATCH_SIZE=128 LR=0.005 ALPHA=0.5 GAMMA=0.3 DEVICE=cuda bash run.sh
```
 
| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | MNIST | Dataset: `MNIST`, `FashionMNIST`, or `CIFAR10` |
| `ROUNDS` | 20 | Number of FL rounds |
| `LOCAL_EPOCHS` | 3 | Local training epochs per round |
| `BATCH_SIZE` | 64 | DataLoader batch size |
| `LR` | 0.01 | Learning rate (SGD with momentum=0.9) |
| `ALPHA` | 0.5 | Dirichlet concentration (lower = more non-IID, very high = equal split) |
| `GAMMA` | 0.3 | EMD weight in peer selection (higher = prefer diverse peers) |
| `THETA` | 0.02 | OCD-FL selector L2 regularization |
| `DEVICE` | cuda | `cuda` or `cpu` |
 
### CLI Arguments (for `main.py`)
 
```bash
python3 main.py \
    --node-id sutdjetson1 \
    --listen 0.0.0.0:50051 \
    --self-ip 192.168.1.86 \
    --peers sutdjetson2=192.168.1.91:50051 sutdjetson3=192.168.1.92:50051 \
    --dataset FashionMNIST \
    --rounds 30 \
    --local-epochs 5 \
    --batch-size 128 \
    --lr 0.005 \
    --dirichlet-alpha 0.5 \
    --selector-theta 0.02 \
    --selector-gamma 0.3 \
    --device cuda \
    --sync-barrier-timeout 30.0
```

## Adding More Jetsons

No code changes needed. Just:

1. Clone the repo onto the new Jetson
2. Run `pip install -r requirements.txt`
3. Run `bash run.sh`

The node auto-discovers all existing peers on the subnet. The Dirichlet data partitioner automatically splits MNIST across however many nodes are listed.

## Networking Notes

- **Port**: All nodes use port `50051` (configurable in `run.sh`)
- **Docker**: `--net=host` means the container shares the host's network stack directly — no port mapping needed
- **Changing networks**: IPs are detected at launch time, so switching between hotspot and WiFi router just requires restarting `run.sh`
- **Firewall**: Ensure port 50051 is not blocked between Jetsons

## Key Design Decisions

### gRPC over raw sockets
gRPC provides automatic serialization, flow control, and deadline propagation. The 256 MB message limit accommodates large models. Protobuf encoding of numpy arrays is ~2x more compact than pickle.

### Asynchronous model exchange
The simulator enforces lock-step rounds. In physical deployment, network latency and heterogeneous compute make synchronous rounds impractical. Instead, each node:
1. Trains locally
2. Discovers who's alive (Ping)
3. Fetches metadata (ExchangeMeta) for knowledge gain
4. Runs the OCD-FL optimizer to select peers
5. Pushes its model to selected peers
6. Waits briefly, then FedAvg-aggregates whatever has arrived

### OCD-FL peer selection with EMD-aware diversity
The `select_peers()` method in `PhysicalClient` reimplements the optimization from `EfficientPeerSelector`, extended with data-heterogeneity awareness:
- Knowledge gain: `KG = (1 - exp(-2d)) * 1(d>0)` where `d = loss_neighbor - loss_self`
- Data diversity: EMD between local and neighbor label distributions, normalized to [0, 1]
- Communication cost: model transfer time + straggler penalty
- Optimization: Adam maximizing `dot(B, KG + gamma * EMD) / dot(B, cost) + theta * ||w||_2` with sigmoid masking
- `gamma` controls how aggressively nodes seek peers with different data distributions — higher values accelerate convergence under non-IID partitions

### Data Distribution Difference (EMD)
Each node computes its local label frequency vector and shares it via `ExchangeMeta`. The Earth Mover Distance `L(D_Vk, D_G) = sum_j |p_j - q_j|` is computed between the local node and each neighbor. This is directly integrated into the peer selection reward function, aligning the implementation with the OCD-FL multi-objective cost function `min O(K) = gamma_1 * O_C + gamma_2 * O_T + gamma_3 * O_L` from the project's theoretical framework.
