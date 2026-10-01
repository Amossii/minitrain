# Step 5: Explicit single-device training loop

One optimizer update has five visible state transitions:

```text
optimizer.zero_grad(set_to_none=True)
logits = model(input_ids)
loss = cross_entropy(logits.reshape(B*T, V), labels.reshape(B*T))
loss.backward()
optimizer.step()
```

Forward creates activations and an autograd graph without changing parameters.
Backward populates each parameter's `.grad`. `optimizer.step()` reads gradients,
creates or updates AdamW moment state, and modifies parameters. The next call to
`zero_grad(set_to_none=True)` prevents accidental gradient accumulation.

The dataset already shifts targets by one token, so loss directly aligns
`logits[b,t]` with `labels[b,t]`. No second shift is performed in the loop.

## Run

Local correctness run:

```bash
conda run -n sglang python scripts/train_single.py \
  --model tiny --device cpu --local-batch-size 2 --seq-len 16 --steps 3
```

Kaggle single-GPU run:

```bash
python3 scripts/train_single.py \
  --model tiny --device cuda --local-batch-size 2 --seq-len 128 --steps 5
```

This step deliberately uses FP32 and reports only loss. CUDA timing, warmup,
throughput, memory metrics, mixed precision, and CSV output belong to the next
metrics/baseline steps.
