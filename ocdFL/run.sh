#!/bin/bash
set -e
cd /app/sutdDFL/ocdFL

bash compile_protos.sh

NODE_ID=$(hostname | tr '[:upper:]' '[:lower:]' | grep -oE '[a-z]+[0-9]+')
if [ -z "$NODE_ID" ]; then
    NODE_ID="jetson_$(hostname -I | awk '{print $1}' | awk -F. '{print $4}')"
fi

MY_IP=$(hostname -I | awk '{print $1}')
SUBNET=$(echo "$MY_IP" | cut -d. -f1-3)
PORT=50051

echo "Node: $NODE_ID"
echo "My IP: $MY_IP"
echo "Scanning ${SUBNET}.0/24 for peers on port $PORT..."

# Retry scan up to 5 times to wait for all Jetsons to come online
MAX_RETRIES=5
RETRY_DELAY=5
PEERS=""

for attempt in $(seq 1 $MAX_RETRIES); do
    PEERS=""
    for i in $(seq 1 254); do
        ip="${SUBNET}.${i}"
        [ "$ip" = "$MY_IP" ] && continue
        if timeout 0.3 bash -c "echo > /dev/tcp/$ip/$PORT" 2>/dev/null; then
            PEER_NAME="jetson_$(echo $ip | awk -F. '{print $4}')"
            PEERS="${PEERS} ${PEER_NAME}=${ip}:${PORT}"
            echo "  Found peer: $PEER_NAME @ $ip"
        fi
    done

    PEER_COUNT=$(echo $PEERS | wc -w)
    EXPECTED_PEERS="${EXPECTED_PEERS:-1}"
    echo "Attempt $attempt/$MAX_RETRIES: Found $PEER_COUNT peer(s), expecting $EXPECTED_PEERS"

    if [ "$PEER_COUNT" -ge "$EXPECTED_PEERS" ]; then
        break
    fi

    echo "Waiting ${RETRY_DELAY}s before retry..."
    sleep $RETRY_DELAY
done

if [ -z "$PEERS" ]; then
    echo "WARNING: No peers found. Starting standalone."
fi

echo ""
echo "Launching: $NODE_ID with peers:$PEERS"
echo "==========================================="

python3 main.py \
    --node-id "$NODE_ID" \
    --listen "0.0.0.0:$PORT" \
    --self-ip "$MY_IP" \
    --peers $PEERS \
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
    --sync-barrier-timeout "${BARRIER_TIMEOUT:-30}"