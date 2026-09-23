#!/bin/bash
# Usage: bash run_all.sh [rounds] [local_epochs]
ROUNDS=${1:-10}
EPOCHS=${2:-2}
mkdir -p logs

python server.py --rounds "$ROUNDS" --clients 4 > logs/server.log 2>&1 &
SERVER_PID=$!
sleep 10   # server ready hote dao

for i in 1 2 3 4; do
  python client.py --cid $i --epochs "$EPOCHS" > logs/client_$i.log 2>&1 &
done

echo "Running... progress dekhte: tail -f logs/server.log"
wait $SERVER_PID
echo "Done. results.csv ar global_model.pt dekho."