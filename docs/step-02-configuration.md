# Step 2: Typed experiment configuration

Every execution strategy must receive the same model shape and training
workload if benchmark comparisons are to be meaningful. `ModelConfig` and
`TrainConfig` provide this shared source of truth and reject invalid values
before GPU state is allocated.

Model and training settings are separate: selecting `medium` changes the model
workload but does not silently change the learning rate or batch size. The
training configuration calls batch size `local_batch_size` deliberately.
During future data-parallel runs:

```text
global_batch_size = local_batch_size * world_size
```

Gradient accumulation is not included yet because it is outside the current
step; when introduced, it will add another explicit factor to that equation.

## Inspect configurations

```bash
python3 scripts/show_config.py --model tiny
python3 scripts/show_config.py --model medium --local-batch-size 4 --precision fp16
```

## Correctness checkpoint

```bash
python3 -m unittest discover -s tests -v
```

The tests verify preset invariants, attention head divisibility, immutability,
invalid input handling, and local/global batch size semantics.
