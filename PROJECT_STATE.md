Phase A：训练框架基础
Step	要做的事情	最终实现功能
1	建项目骨架、Git、Kaggle 环境检查	可复现开发环境
2	实现配置系统	模型/训练参数统一管理
3	实现 Synthetic Dataset	可控的训练 workload
4	手写 Decoder-only Transformer	自己可修改的训练模型
5	实现单 GPU 训练	Single-GPU baseline
6	加指标和 CSV	step time / tok/s / memory
7	做单卡 benchmark	得到可靠 baseline


Step 1：项目初始化
先建立：
MiniTrain/
├── minitrain/
├── scripts/
├── configs/
├── tests/
├── results/
├── docs/
└── README.md

要做：
检查 PyTorch
检查 CUDA
检查 GPU 数量
检查 NCCL
打印 GPU 型号
打印 GPU topology

实现一个：
python scripts/check_env.py

输出类似：
PyTorch: ...
CUDA: ...
NCCL: ...
GPU count: 2
GPU 0: Tesla T4
GPU 1: Tesla T4

这个 Step 不做任何训练。
完成标准：
Kaggle clone repo
→ 安装依赖
→ check_env.py 成功

同时从第一天开始 Git 提交代码，不要长期只存在 /kaggle/working。
Step 2：配置系统
实现：
ModelConfig
TrainConfig

例如：
hidden_size
num_layers
num_heads
intermediate_size
vocab_size
seq_len

batch_size
lr
steps
precision

支持：
tiny
small
medium

目标是以后所有训练方式：
Single GPU
DDP
FSDP2
DeepSpeed

都使用同一份模型配置。
否则后面的 benchmark 很容易比较错。
Step 3：Synthetic Dataset
先不要下载真实数据集。
生成：
input_ids
[B, T]

labels
[B, T]

随机 token。
需要支持：
batch_size
seq_len
vocab_size
seed

这样以后：
DDP vs FSDP2 vs ZeRO

比较的是训练系统，而不是 DataLoader。
Step 4：手写 Mini Transformer
实现：
Embedding
↓
TransformerBlock × N
↓
RMSNorm
↓
LM Head

Block：
RMSNorm
Attention
Residual
RMSNorm
SwiGLU MLP
Residual

这里暂时：
不做 TP
不做 FlashAttention
不做 fancy optimization

目标只有：
得到一个以后可以自由拆 Column / Row Parallel 的 Transformer。

需要有单元测试：
输入:
[B,T]

输出:
[B,T,V]

Step 5：单 GPU Training Loop
自己写：
optimizer.zero_grad()loss = model(...)loss.backward()optimizer.step()


实现：
forward
loss
backward
optimizer step

不要使用：
Trainer
Lightning
Accelerate

这一阶段必须清楚控制训练生命周期。
完成：
python scripts/train_single.py

能看到：
step=0 loss=...
step=1 loss=...
...

Step 6：Metrics
加入统一指标：
step_time
samples/s
tokens/s

peak_memory_allocated
peak_memory_reserved

forward_time
backward_time
optimizer_time

结果保存：
results/raw/*.csv

例如：
strategy,gpus,batch,seq_len,step_time,tokens_per_sec,peak_memory
single,1,8,512,...

从这里开始建立统一 benchmark schema。
后面所有实现都往这个 CSV 写。
Step 7：Single GPU Baseline
正式跑：
Tiny
Small
Medium

记录：
模型参数量
GPU memory
step time
tokens/s

并做：
warmup
正式 measurement
固定 seed

最终得到项目第一张 baseline 表。
到这里，你已经有一个完整但单卡的 Training Systems 项目骨架。
Phase B：Distributed Communication
Step	要做的事情	实现功能
8	学会 torchrun 初始化	2 GPU Process Group
9	Collective correctness	AllReduce / AllGather / ReduceScatter
10	Collective benchmark	latency / bandwidth
11	实现 DDP	真正双卡训练
12	DDP correctness	验证参数同步
13	Strong/Weak Scaling	DDP scaling benchmark
14	DDP Profiler	通信计算 overlap


这一阶段做完已经可以开始投第一批 LLM Infra 实习。
Step 8：Distributed Runtime
实现：
init_distributed()
cleanup_distributed()

理解：
RANK
LOCAL_RANK
WORLD_SIZE

Kaggle：
torchrun \
  --standalone \
  --nproc-per-node=2 \
  scripts/distributed_hello.py

要求输出：
rank=0 local_rank=0 world_size=2
rank=1 local_rank=1 world_size=2

并确保：
rank0 → cuda:0
rank1 → cuda:1

这一步暂时不要训练模型。
Step 9：Collective Correctness
自己写几个最小实验：
broadcast
reduce
all_reduce
all_gather
reduce_scatter
barrier

例如：
rank0 tensor = [1]
rank1 tensor = [2]

AllReduce(SUM)

rank0 → [3]
rank1 → [3]

重点不是 API，而是搞清楚：
输入 shape
输出 shape
数据最后在哪
通信发生了什么

这一步会成为后面：
DDP
FSDP
TP

的基础。
Step 10：NCCL Collective Benchmark
实现：
scripts/benchmark_collectives.py

测试：
1 KB
4 KB
16 KB
...
256 MB

Collective：
AllReduce
AllGather
ReduceScatter

输出：
message_size
latency_ms
bandwidth_GBps

做第一张正式结果图：
Bandwidth
   ^
   |
   |       ________
   |     /
   |____/
   +-------------> message size

这是非常值得展示的 Infra 内容。
Phase C：DDP
Step 11：实现 DDP Training
把 Step 5 的代码改造成：
torchrun
↓
init_process_group
↓
set_device(local_rank)
↓
DistributedSampler
↓
DDP(model)
↓
training

注意：
一个进程一张 GPU

此时：
每卡都有完整模型
每卡计算自己的 batch
backward 时同步梯度

Step 12：验证 DDP Correctness
不要只证明“能跑”。
验证：
rank0 参数
==
rank1 参数

训练前后都检查。
再检查：
global batch
=
local batch × world size

最好加入一个故意关闭同步的实验，让两卡模型逐渐不同。
这样真正理解：
为什么 Data Parallel 必须同步 gradient。

Step 13：Strong / Weak Scaling
Strong：
global batch 固定

1 GPU:
batch=256

2 GPU:
batch=128/GPU

Weak：
local batch 固定

1 GPU:
batch=128

2 GPU:
global batch=256

记录：
throughput
step time
speedup
scaling efficiency

重点不是必须证明：
2 GPU = 2x

而是解释为什么没有 2x。
Step 14：DDP Profiler
加入：
torch.profiler
record_function

观察：
forward GEMM
backward GEMM
ncclAllReduce
optimizer

目标找到：
backward compute
        ↘
         NCCL AllReduce

之间的 overlap。
这一阶段应该真正理解：
gradient bucket
bucket ready
AllReduce
communication overlap

到这里，项目 V1 完成。
简历已经可以写：
PyTorch DDP
NCCL
Collectives
Strong/Weak Scaling
torch.profiler

Phase D：FSDP2
Step	内容
15	FSDP2 correctness
16	Memory benchmark
17	OOM boundary
18	FSDP2 profiling


Step 15：接入 FSDP2
保持：
同一个 Transformer
同一个 dataset
同一个 training loop

只改变分布式策略。
实现：
fully_shard(...)

重点观察：
parameter shard
AllGather
ReduceScatter

Step 16：DDP vs FSDP2 Memory
比较：
DDP
FSDP2

测：
parameters
gradients
optimizer state
peak memory

同时自己理论估算。
例如：
Parameter Memory
Gradient Memory
Adam States
Activation

然后和：
torch.cuda.max_memory_allocated()


比较。
Step 17：OOM Boundary
逐步增大模型：
100M
200M
300M
400M
...

直到：
DDP → OOM
FSDP2 → PASS

记录：
Maximum trainable model size

这会成为项目非常重要的一张图。
Step 18：Profile FSDP2
观察 timeline：
AllGather
GEMM
ReduceScatter

回答：
FSDP 为什么省显存？
为什么会增加通信？
为什么可能比 DDP 慢？

到这里是 V2。
这个版本已经很适合投分布式训练 / AI Infra 岗。
Phase E：Tensor Parallel
Step 19：ColumnParallelLinear
自己实现：
Linear:
H → 4H

把 output dimension 拆到两卡：
GPU0:
W[:, :2H]

GPU1:
W[:, 2H:]

实现：
local matmul
AllGather（需要完整输出时）

然后和普通：
nn.Linear


结果对齐。
Step 20：RowParallelLinear
把 input dimension 拆开。
实现：
GPU0:
X0 @ W0

GPU1:
X1 @ W1

然后：
AllReduce

得到完整结果。
也必须和普通 Linear 数值对齐。
Step 21：Transformer TP
把：
QKV
Gate
Up

做成 Column Parallel。
把：
Attention Output
Down

做成 Row Parallel。
最终：
torchrun --nproc-per-node=2 scripts/train_tp.py

能够训练。
这里不追求复刻 Megatron。
重点是亲手实现：
Column Parallel
Row Parallel
AllReduce
AllGather

理解模型并行的数据流。
Phase F：DeepSpeed
Step	内容
22	ZeRO-1 / 2 / 3
23	Unified Benchmark
24	Checkpoint + README + 最终项目


Step 22：DeepSpeed ZeRO
这时才接：
ZeRO-1
ZeRO-2
ZeRO-3

分别跑通。
重点不是配置文件。
而是能回答：
ZeRO-1 分了什么？
ZeRO-2 多分了什么？
ZeRO-3 又多分了什么？

因此：
memory 怎么变化？
communication 怎么变化？

Step 23：统一 Benchmark
这是整个项目最终核心。
统一：
模型
dataset
global batch
seq_len
precision
optimizer
steps

比较：
Strategy	Peak Memory	Step Time	Tokens/s	Comm
Single				
DDP				
FSDP2				
ZeRO-1				
ZeRO-2				
ZeRO-3				


TP 单独比较：
Single
vs
TP=2

因为它和 DP/FSDP 解决的问题不同。
Step 24：Checkpoint + 最终封板
最后加入：
save checkpoint
↓
停止训练
↓
reload
↓
继续训练

至少测试：
Single GPU
DDP
FSDP2

如果时间够，再加入 DeepSpeed checkpoint。
最终生成：
NCCL bandwidth 图

DDP strong scaling 图

DDP weak scaling 图

DDP/FSDP/ZeRO memory 图

DDP/FSDP/ZeRO throughput 图

maximum trainable model size 图

Profiler timeline

README 首页最终应该直接呈现：
MiniTrain

Architecture
↓
Benchmark Results
↓
Memory Comparison
↓
Profiler
↓
Technical Analysis

而不是先写几千字理论。
整个项目的能力闭环
完成 24 Step 后，你应该可以从头解释：
GPU0            GPU1
 |               |
 |---- NCCL -----|
        ↓
Collectives
        ↓
DDP
        ↓
Gradient AllReduce
        ↓
FSDP2
        ↓
Parameter AllGather
Gradient ReduceScatter
        ↓
Tensor Parallel
        ↓
Column / Row Parallel
        ↓
DeepSpeed
        ↓
ZeRO-1/2/3

并且不只是解释，还能：
实现
↓
验证 correctness
↓
benchmark
↓
profile
↓
发现瓶颈
↓
解释结果

这才是这个项目对找实习真正有价值的地方。
建议我们之后就和 MiniServe 一样，严格按 Step 1 → Step 24 一步一步完成，每一步都采用：
1. 本 Step 要解决什么问题
2. 原理
3. 要创建/修改哪些文件
4. 完整代码
5. Kaggle 执行命令
6. 预期输出
7. 如何验证正确
8. 常见问题
9. Checkpoint
10. Git commit
11. 这一 Step 的面试价值