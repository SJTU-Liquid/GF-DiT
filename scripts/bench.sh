python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend v1/videos --port 8098 \
    --dataset random --task t2v \
    --num-prompts 100 --max-concurrency 100 --request-rate inf \
    --width 480 --height 832 --num-frames 81 \
    --num-inference-steps 50 \
    --warmup-requests 2 --warmup-num-inference-steps 2 \
    --seed 42 \
    --output-file /tmp/bench_elastic.json