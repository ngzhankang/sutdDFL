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
| Data Distribution Difference | Not explicitly used | EMD computed over label distributions via metadata |
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

- Docker container `sutd-dfl-jetson:v1` launched with `--net=host`
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

<img width="1178" height="153" alt="image" src="https://github.com/user-attachments/assets/9945cc89-d2f3-47e7-9829-e6ffcd72daae" />

### 3. Override defaults with environment variables

```bash
ROUNDS=50 LOCAL_EPOCHS=5 BATCH_SIZE=128 LR=0.005 ALPHA=0.1 DEVICE=cpu bash run.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUNDS` | 20 | Number of FL rounds |
| `LOCAL_EPOCHS` | 3 | Local training epochs per round |
| `BATCH_SIZE` | 64 | DataLoader batch size |
| `LR` | 0.01 | Learning rate |
| `ALPHA` | 0.5 | Dirichlet concentration (lower = more non-IID) |
| `DEVICE` | cuda | `cuda` or `cpu` |

### 4. Check results

Metrics are saved to `./logs/<node_id>_metrics.json` on each Jetson, e.g.:

```bash
cat logs/sutdjetson1_metrics.json | python3 -m json.tool
```

Each entry contains per-round train loss, test accuracy (pre/post aggregation), number of peers selected, and round time.

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

### OCD-FL peer selection preserved
The `select_peers()` method in `PhysicalClient` reimplements the exact optimization from `EfficientPeerSelector`:
- Knowledge gain: `KG = (1 - exp(-2d)) * 1(d>0)` where `d = loss_neighbor - loss_self`
- Communication cost: model transfer time + straggler penalty
- Optimization: Adam maximizing `(B*KG) / (B*cost) + theta*||w||_2` with sigmoid masking

### Data Distribution Difference (EMD)
Each node computes its local label frequency vector and shares it via `ExchangeMeta`. The Earth Mover Distance `L(D_Vk, D_G) = sum_j |p_j - q_j|` is available for extended peer selection criteria.
