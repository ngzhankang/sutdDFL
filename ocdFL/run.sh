# Remove the entire scan loop and just launch directly
echo ""
echo "Launching: $NODE_ID (standalone start, peers discovered each round)"
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