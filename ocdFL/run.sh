#!/bin/bash
set -e
cd /app/sutdDFL/ocdFL

bash compile_protos.sh

NODE_ID=$(hostname | tr '[:upper:]' '[:lower:]' | grep -oE '[a-z]+[0-9]+')
if [ -z "$NODE_ID" ]; then
    NODE_ID="jetson_$(hostname -I | awk '{print $1}' | awk -F. '{print $4}')"
fi

MY_IP=$(hostname -I | awk '{print $1}')
PORT=50051

echo "Node: $NODE_ID"
echo "My IP: $MY_IP"
echo "Total nodes in cluster: ${TOTAL_NODES:-3}"
echo "==========================================="

python3 main.py \
    --node-id "$NODE_ID" \
    --listen "0.0.0.0:$PORT" \
    --self-ip "$MY_IP" \
    --total-nodes "${TOTAL_NODES:-3}" \
    --rounds "${ROUNDS:-20}" \
    --local-epochs "${LOCAL_EPOCHS:-3}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --lr "${LR:-0.01}" \
    --dirichlet-alpha "${ALPHA:-0.5}" \
    --device "${DEVICE:-cuda}" \
    --data-dir ./data \
    --log-dir ./logs \
    --dataset "${DATASET:-MNIST}" \
    --selector-gamma "${GAMMA:-0.3}" \
    --selector-theta "${THETA:-0.02}" \
    --sync-barrier-timeout "${BARRIER_TIMEOUT:-60}"