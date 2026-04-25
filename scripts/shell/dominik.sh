#!/usr/bin/env bash

PORTS=(8000 8001 8002)

for port in "${PORTS[@]}"; do
python scripts/dominik.py --port "$port" &
sleep 10
done


wait
