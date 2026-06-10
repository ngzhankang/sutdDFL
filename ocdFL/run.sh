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

# Build --peers args from PEER_IPS env var (space-separated IPs, e.g. "10.32.2.198 10.32.5.12")
# When set, the subnet scan is skipped and these IPs are used directly.
PEERS_ARGS=""
if [ -n "$PEER_IPS" ]; then
    echo "Using explicit peer IPs: $PEER_IPS"
    for ip in $PEER_IPS; do
        peer_id="jetson_$(echo "$ip" | tr '.' '_')"
        PEERS_ARGS="$PEERS_ARGS --peers ${peer_id}=${ip}:${PORT}"
    done
else
    echo "No PEER_IPS set — will fall back to subnet scan"
fi
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
    --device "${DEVICE:-cpu}" \
    --data-dir ./data \
    --log-dir ./logs \
    --dataset "${DATASET:-FashionMNIST}" \
    --selector-gamma "${GAMMA:-0.3}" \
    --selector-theta "${THETA:-0.02}" \
    --sync-barrier-timeout "${BARRIER_TIMEOUT:-60}" \
    $PEERS_ARGS
