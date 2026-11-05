| Configuration | Added mechanism | SLO (%) | Mean (s) |
|---|---|---:|---:|
| Legacy (SP4) | --- | 27.9 | 471.2 |
| FCFS-SP1 | single-GPU lanes | 55.7 | 174.8 |
| SJF-SP1 | request-size ordering | 57.4 | 164.9 |
| SRTF-SP1 | step-level preemption | 70.5 | 154.9 |
| EDF-SP1 | deadline ordering | 83.6 | 143.7 |
| EDF | dynamic parallelism | 85.2 | 121.5 |
| EDF-SP1 @40% load |  | 100.0 | 81.3 |
| EDF @40% load |  | 100.0 | 50.5 |
| FCFS-SP4 @40% load |  | 80.0 | 70.1 |
