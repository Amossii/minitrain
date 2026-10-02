# Step 8: torchrun and process-group runtime

`torchrun` launches one Python process per worker and injects rendezvous plus
identity variables. On a single node with two GPUs:

```text
process 0: RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 -> cuda:0
process 1: RANK=1 LOCAL_RANK=1 WORLD_SIZE=2 -> cuda:1
```

`RANK` identifies a process globally. `LOCAL_RANK` identifies it within the
current node and is therefore the value used for CUDA device mapping.
`WORLD_SIZE` is the total number of participating processes.

The default process group is shared global runtime state. Initialization uses
`env://`, which consumes `MASTER_ADDR` and `MASTER_PORT` supplied by torchrun.
Cleanup destroys this state before process exit.

## Local CPU correctness run

```bash
conda run -n sglang python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  scripts/distributed_hello.py --backend gloo
```

## Kaggle two-GPU NCCL run

```bash
python3 scripts/check_env.py --require-gpus 2
torchrun --standalone --nproc-per-node=2 \
  scripts/distributed_hello.py --backend nccl
```

Expected identities are rank/local-rank pairs `(0,0)` and `(1,1)`, both with
world size 2. Output order is not guaranteed because processes run concurrently.
This step establishes connectivity only; collective tensor semantics begin in
Step 9.
