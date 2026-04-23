# sutdDFL — Decentralised Federated Learning on NVIDIA Jetson Orin Nano

A research project by the [Temasek Laboratories @ SUTD](https://www.sutd.edu.sg/) exploring Decentralised Federated Learning (DFL) deployed physically across a cluster of NVIDIA Jetson Orin Nano edge devices.

> **Full setup instructions, hardware configuration, and implementation details are documented in the [Wiki](https://github.com/ngzhankang/sutdDFL/wiki).**

---

## Overview

Unlike centralised Federated Learning, this project implements **peer-to-peer model aggregation** across edge devices — no central server required. Each Jetson node trains locally on its own data partition and selectively exchanges model updates with neighbours based on the **OCD-FL (Opportunistic Communication-efficient Decentralised FL)** knowledge gain framework.

Key properties:
- **No single point of failure** — fully decentralised topology
- **Privacy-preserving** — only model weights are transmitted, never raw data
- **Resource-aware** — peer selection accounts for data distribution differences (Earth Mover's Distance) and device computation costs
- **Edge-native** — designed for ARM64 Jetson hardware with CUDA acceleration

---

## Hardware & Environment

| Component | Spec |
|---|---|
| Device | NVIDIA Jetson Orin Nano Dev Kit |
| JetPack | 6.2.1 |
| CUDA | 12.6 |
| OS | Ubuntu 22.04.5 LTS |
| Storage | 128GB A2 microSD (U3/V30, A1/A2) |

---

## Repository Structure

```
sutdDFL/
└── ocdFL/          # OCD-FL algorithm implementation (see ocdFL/README.md)
```

---

## Docker Image

The project runs inside a Docker container (`sutd-dfl-jetson:v1`) built on NVIDIA's JetPack-optimised PyTorch base image. The image is publicly available on GitHub Container Registry — no login required.

**Pull the image on any Jetson:**
```bash
docker pull ghcr.io/ngzhankang/sutd-dfl-jetson:v1
```

**Run the container:**
```bash
sudo docker run --runtime nvidia --net=host -v /home/$USER/SUTD:/app sutd-dfl-jetson:v1
```

For full Docker setup instructions including microSD configuration and NVIDIA Container Toolkit installation, see the [Getting Started](https://github.com/ngzhankang/sutdDFL/wiki/Getting-Started) wiki page.

---

## Implementations

See the [Implementations](https://github.com/ngzhankang/sutdDFL/wiki/Implementations) wiki page for details on experiments and algorithm variants.

For the OCD-FL specific implementation, refer to [`ocdFL/README.md`](./ocdFL/README.md).

---

## Team

| Name | Role |
|---|---|
| Prof. Marie Therese Siew | Principal Investigator |
| Lucas Liew | Researcher |
| Skylar | Researcher |
| Ng Zhan Kang | Researcher |

SUTD 2025
