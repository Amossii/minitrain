2×T4, FP32 AdamW, local batch size=1, seq_len=128

DDP:
largest verified PASS ≈ 601M
peak allocated ≈ 13.52 GiB
≈ 24.15 B/parameter

FSDP2:
largest verified PASS ≈ 1.403B
peak allocated ≈ 13.10 GiB
≈ 10.03 B/parameter

verified model-capacity gain:
≈ 2.33×