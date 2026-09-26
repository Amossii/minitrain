# Step 1: Project skeleton and environment check

The checker establishes whether the runtime can support later CUDA and NCCL
experiments. PyTorch reports the software view of CUDA devices and NCCL, while
`nvidia-smi topo -m` reports the physical connectivity relevant to collective
communication.

No performance conclusion should be drawn from this step. It performs no GPU
workload and records no benchmark measurements.

## Kaggle checkpoint

From `/kaggle/working/MiniTrain`, run:

```bash
python3 scripts/check_env.py --require-gpus 2
python3 -m unittest discover -s tests -v
```

The first command is correct only when it prints two GPU names and ends with
`Environment validation: PASSED`. Save the actual output with the experiment
notes; do not substitute example hardware names for observed results.

