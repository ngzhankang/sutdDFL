#!/bin/bash
# run_jetson2.sh — Launch on Jetson 2 (192.168.38.209)

set -e
cd /app

bash compile_protos.sh

python main.py \
    --node-id jetson2 \
    --listen 0.0.0.0:50051 \
    --self-ip 192.168.38.209 \
    --peers jetson1=192.168.38.37:50051 \
    --rounds 20 \
    --local-epochs 3 \
    --batch-size 64 \
    --lr 0.01 \
    --dirichlet-alpha 0.5 \
    --device cuda \
    --data-dir ./data \
    --log-dir ./logs
