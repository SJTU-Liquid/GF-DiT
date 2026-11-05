#!/bin/bash
# usage: run_abort_test.sh <comma-separated GPUs> <abort|watchdog> <out_dir> [self|external]
set -u
GPUS=$1; MODE=$2; OUT=$3; KILL=${4:-self}; rm -rf "$OUT"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPUS MODE KILL RUN_DIR=$OUT MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + RANDOM % 400))
export WORLD_SIZE=$(echo $GPUS | tr ',' '\n' | wc -l)
[ "$MODE" = watchdog ] && export GFC_ABORT_CHECK=0 || export GFC_ABORT_CHECK=64
for r in $(seq 0 $((WORLD_SIZE - 1))); do
  RANK=$r LOCAL_RANK=$r timeout 240 "${PYTHON:-python3}" "$(dirname "$0")/abort_test.py" > "$OUT/rank$r.log" 2>&1 &
done
if [ "$KILL" = external ]; then
  until [ -f "$OUT/looping" ]; do sleep 0.05; done
  sleep "$(python3 -c 'import random; print(round(random.uniform(0.5, 1.5), 3))')"
  python3 -c "import time; open('$OUT/death_time','w').write(repr(time.time()))"
  kill -9 "$(cat "$OUT/pid$((WORLD_SIZE - 1))")"
fi
wait
grep -h RESULT "$OUT"/rank*.log | sed 's/^RESULT //'
