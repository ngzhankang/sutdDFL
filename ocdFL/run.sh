cat > run.sh << 'EOF'
#!/bin/bash
set -e
cd /app/sutdDFL/ocdFL

bash compile_protos.sh

# Auto-detect identity from hostname (e.g. sutdJetson1 → jetson1)
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

PEERS=""
for i in $(seq 1 254); do
    ip="${SUBNET}.${i}"
    [ "$ip" = "$MY_IP" ] && continue
    if timeout 0.3 bash -c "echo > /dev/tcp/$ip/$PORT" 2>/dev/null; then
        # Query the peer's node_id via a quick gRPC ping, fallback to IP-based name
        PEER_NAME="jetson_$(echo $ip | awk -F. '{print $4}')"
        PEERS="${PEERS} ${PEER_NAME}=${ip}:${PORT}"
        echo "  Found peer: $PEER_NAME @ $ip"
    fi
done

if [ -z "$PEERS" ]; then
    echo "WARNING: No peers found yet. Starting as standalone (will retry discovery each round)."
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
    --log-dir ./logs
EOF
chmod +x run.sh
