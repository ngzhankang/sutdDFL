#!/bin/bash
# run_jetson1.sh — Launch on Jetson 1 (192.168.38.37)
# Execute inside the Docker container: docker exec -it <container> bash
# Or directly on host if deps are installed.

set -e

cd /app/sutdDFL/ocdFL  # mapped from /home/$USER/SUTD

# Compile protos (only needed once)
bash compile_protos.sh

python main.py \
    --node-id jetson1 \
    --listen 0.0.0.0:50051 \
    --self-ip 192.168.38.37 \
    --peers jetson2=192.168.38.209:50051 \
    --rounds 20 \
    --local-epochs 3 \
    --batch-size 64 \
    --lr 0.01 \
    --dirichlet-alpha 0.5 \
    --device cuda \
    --data-dir ./data \
    --log-dir ./logs
