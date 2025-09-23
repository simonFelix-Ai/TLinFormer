# 设计哲学

不想牺牲任何历史信息（Full Context Awareness），不牺牲注意力计算的精确性（Exact Attention），仍然希望获得线性的计算复杂度（Linear Complexity).

TxFormer没有走主流的“近似 Softmax”（Kernel Methods）或“丢弃 Token”（Sparse Methods）的老路，而是回归到神经网络的“连接拓扑”层面，通过重构信息流来解决效率问题。

有意引入信息瓶颈，以强制的压缩换取理论上的极致效率。

因为是端到端训练，所以其性能，会优于专门对KV Cache进行额外压缩的模型。

# 说明

本仓库包括了 TLinFormer 和 TconstFormer 两个模型架构的源码。

更多模型的性能表现与差别，请查看论文：https://arxiv.org/abs/2508.20407, https://arxiv.org/abs/2509.00202

TLinFormer: 严格的O(N) KV Cache消耗（但也远远优于标准的自解码模型）， 严格的O(N)注意力计算消耗。
TconstFormer: 严格的O(1) KV Cache消耗，摊销的O(1)注意力计算消耗。

应用场景：
对于一般场景：TLinFormer因为优化没有那么极端，TLinFormer可能更适合，尽管论文中测试TconstFormer解码性能更好，但没有在更大规模参数下验证。
TconstFormer因资源消耗极低，特别适合端侧，比如机器人等。

# 代码库结构

```
.
├── configs/                # 配置文件目录
│   └── base_config.yaml    # 基础配置文件
├── dataset_cache/			# 数据集相关
├── results/                # 存放训练日志、模型权重等
│   └── runs/               # TensorBoard 日志
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载和预处理
│   ├── models/             # 模型定义
│   │   ├── baseline.py     # 基线模型（标准Transformer）
│   │   └── my_variant.py   # 模型变体
│   └── utils               # 工具函数，如日志记录器
├── tests/                  # 测试相关
│   ├── test_xxx.py         # 测试用例
├── requirements.txt        # Python依赖
├── train.py                # 训练脚本
└── readme.md               # 项目说明
```

---

# 使用方法

## 1. 环境配置

python使用3.12


```bash
conda create -n TxFormer python=3.12
conda activate  TxFormer
pip install -r requirements.txt
```

## 2. 配置文件

在 `configs/` 目录下创建你的实验配置，例如 `my_model_1b.yaml`。可以从 `base_config.yaml` 复制并修改。关键配置项包括：

## 3. 开始训练

```bash
python train.py --config configs/my_model_1b.yaml
```

或者执行根目录的 build.sh, 训练所有配置。

## 4. 测试

```bash
python -m tests.test_generate_performance --config configs/TLinLLM_41M_1K.yaml
```

或者执行 tests目录下的test.sh，进行所有配置测试。

## 5. 监控训练状态

训练过程中的所有指标（如损失、学习率）都会被记录到 `results/runs/` 目录下。使用 TensorBoard 查看：

```bash
tensorboard --logdir results/runs
```

然后浏览器打开 `http://localhost:6006` 即可看到实时的损失曲线图。


# 需改进

## 1. 使用moe
## 2. 更大参数涌现能力验证
## 3. 大海捞针等能力验证

# 更多

见doc目录
