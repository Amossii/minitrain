# Step 4: Handwritten decoder-only Transformer

The model maps integer token IDs to one unnormalized score per vocabulary item:

```text
[B,T] token IDs
  -> token embedding + learned position embedding [B,T,C]
  -> TransformerBlock x L                 [B,T,C]
  -> RMSNorm                              [B,T,C]
  -> LM head                              [B,T,V]
```

Each pre-norm block applies:

```text
x = x + CausalSelfAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

## Attention shapes

For `C = H * D`:

```text
Q, K, V projections: [B,T,C]
split heads:          [B,H,T,D]
Q @ K^T:              [B,H,T,T]
causal softmax:        [B,H,T,T]
probabilities @ V:     [B,H,T,D]
merge heads:           [B,T,C]
output projection:     [B,T,C]
```

The lower-triangular causal mask makes score `(query_position, key_position)`
invisible whenever the key lies in the future. A correctness test changes only
future input tokens and verifies that earlier logits remain unchanged.

## Tensor-parallel preparation

Projection boundaries remain explicit rather than hidden behind a large model
framework. Q/K/V and SwiGLU gate/up projections can later be column-sharded;
attention output and SwiGLU down projections can later be row-sharded. Step 4
does not perform any sharding.

## Run

```bash
conda run -n sglang python scripts/inspect_model.py \
  --model tiny --batch-size 2 --seq-len 8
conda run -n sglang python -m unittest discover -s tests -v
```

The inspection's mean-logit backward pass checks graph connectivity only. A
real cross-entropy training objective is intentionally deferred to Step 5.
