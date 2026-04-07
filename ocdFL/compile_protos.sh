#!/bin/bash
# Run from the dfl_physical directory
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./client/transport \
    --grpc_python_out=./client/transport \
    ./protos/dfl.proto

# Fix import path in generated grpc file
sed -i 's/import dfl_pb2/from client.transport import dfl_pb2/' \
    ./client/transport/dfl_pb2_grpc.py

echo "Proto compilation complete."
