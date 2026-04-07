# Physical Decentralized Federated Learning on Jetson Orin Nano

This codebase converts the [nizar-masmoudi/decentralized-federated-learning](https://github.com/nizar-masmoudi/decentralized-federated-learning) simulator into a **real physical P2P deployment** across two Nvidia Jetson Orin Nanos connected via a mobile hotspot.

## Architecture Overview

```
┌─────────────────────────────────┐        gRPC over WiFi        ┌─────────────────────────────────┐
│  Jetson 1  (192.168.38.37)      │◄────────────────────────────►│  Jetson 2  (192.168.38.209)     │
│                                 │   Ping / ExchangeMeta /      │                                 │
│  PhysicalClient("jetson1")      │   PushModel                  │  PhysicalClient("jetson2")      │
│  ├─ LeNet-5 (MNIST)             │                              │  ├─ LeNet-5 (MNIST)             │
│  ├─ gRPC server :50051          │                              │  ├─ gRPC server :50051          │
│  ├─ OCD-FL peer selector        │                              │  ├─ OCD-FL peer selector        │
│  └─ FedAvg aggregator           │                              │  └─ FedAvg aggregator           │
└─────────────────────────────────┘                              └─────────────────────────────────┘
```

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
dfl_physical/
├── main.py                      # Entry point (run on each Jetson)
├── compile_protos.sh            # Compiles .proto → Python gRPC stubs
├── run_jetson1.sh               # Launch script for Jetson 1
├── run_jetson2.sh               # Launch script for Jetson 2
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

## Quick Start

### 1. Install dependencies (inside Docker container)

```bash
pip install grpcio grpcio-tools protobuf
```

### 2. Compile protobuf (run once)

```bash
bash compile_protos.sh
```

### 3. Launch on both Jetsons simultaneously

**Jetson 1:**
```bash
bash run_jetson1.sh
```

**Jetson 2:**
```bash
bash run_jetson2.sh
```

The nodes will wait up to 30 seconds for each other to come online, then begin the FL rounds.

### 4. Check results

Metrics are saved to `./logs/jetson1_metrics.json` and `./logs/jetson2_metrics.json`.

## Key Design Decisions

### gRPC over raw sockets
gRPC provides automatic serialization, flow control, and deadline propagation. The 256 MB message limit accommodates large models. Protobuf encoding of numpy arrays is ~2× more compact than pickle.

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
- Knowledge gain: `KG = (1 - exp(-2δ)) · 𝟙(δ>0)` where `δ = loss_neighbor - loss_self`
- Communication cost: model transfer time + straggler penalty
- Optimization: Adam maximizing `(β·KG) / (β·cost) + θ‖w‖₂` with sigmoid masking

### Data Distribution Difference (EMD)
Each node computes its local label frequency vector and shares it via `ExchangeMeta`. The Earth Mover Distance `L(D_Vk, D_G) = Σ_j |p_j - q_j|` is available for extended peer selection criteria.
