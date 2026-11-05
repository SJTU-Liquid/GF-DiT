Table 1: NCCL subgroup setup (8x NVIDIA H20, torch 2.10.0+cu129, NCCL 2.27.5)

| Group size | new_group (ms) | First coll. (ms) | Warm coll. (ms) | Memory (MB/GPU) |
|---:|---:|---:|---:|---:|
| 2 | 0.77 | 217.3 | 0.19 | 484 |
| 4 | 0.46 | 532.0 | 0.56 | 535 |
| 6 | 0.57 | 663.5 | 0.72 | 535 |
| 8 | 0.57 | 777.7 | 0.89 | 535 |

GFC group registration (2x NVIDIA H20, 16 successive size-2 groups):
- per-group register time: median 60.6 us (min 56.3, first/cold 122.5); paper: ~60 us
- per-group memory: 0 MB after 16 groups (NCCL: ~485-535 MB each)
- one-time runtime init: 1028 ms, 76 MB shared symmetric region
