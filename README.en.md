# Latest Breakthrough

I have successfully overcome the technical bottleneck of training 1 million tokens in a single pass on a single 16GB VRAM GPU. To clarify, this is not a cumulative total of 1M tokens achieved through small segments (e.g., 512 tokens x N); it is a genuine, simultaneous 1-million-token context in one training step.
Due to limited personal funding, I am currently unable to scale this research further. If you are interested in providing support or collaborating on large-scale research, please feel free to reach out.

# Design Philosophy

The goal is to achieve linear computational complexity (Linear Complexity) without sacrificing any historical information (Full Context Awareness) or the precision of attention calculations (Exact Attention).

Instead of following the mainstream paths of "approximating Softmax" (Kernel Methods) or "discarding tokens" (Sparse Methods), TxFormer returns to the level of the neural network's "connection topology," solving the efficiency problem by reconstructing the information flow.

It intentionally introduces an information bottleneck, trading forced compression for theoretically extreme efficiency.

Because it is trained end-to-end, its performance is superior to models that apply separate, additional compression techniques to the KV Cache.


# Notes

This repository includes the source code for two model architectures: TLinFormer and TconstFormer.

For more details on the models' performance and differences, please refer to the papers: https://arxiv.org/abs/2508.20407, https://arxiv.org/abs/2509.00202

TLinFormer: Features strict O(N) KV Cache consumption (though still far superior to standard auto-regressive models) and strict O(N) attention computation cost.

TconstFormer: Features strict O(1) KV Cache consumption and amortized O(1) attention computation cost.

Application Scenarios:

For general scenarios: TLinFormer may be more suitable because its optimizations are less extreme. Although the paper shows TconstFormer achieved better decoding performance in tests, this has not been validated at a larger parameter scale.
Due to its extremely low resource consumption, TconstFormer is particularly well-suited for edge applications, such as in robotics.

# Repository Structure

```
.
├── configs/                # Configuration file directory
│   └── base_config.yaml    # Base configuration file
├── dataset_cache/			# Dataset related
├── results/                # Stores training logs, model weights, etc.
│   └── runs/               # TensorBoard logs
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models/             # Model definitions
│   │   ├── baseline.py     # Baseline model (standard Transformer)
│   │   └── my_variant.py   # Model variant
│   └── utils               # Utility functions, such as loggers
├── tests/                  # Test related
│   ├── test_xxx.py         # Test cases
├── requirements.txt        # Python dependencies
├── train.py                # Training script
└── readme.md               # Project description
```

---

# Usage

## 1. Environment Setup

Use Python 3.12

```bash
conda create -n TxFormer python=3.12
conda activate  TxFormer
pip install -r requirements.txt
```

## 2. Configuration File

Create your experiment configuration in the `configs/` directory, for example `my_model_1b.yaml`. You can copy and modify it from `base_config.yaml`. Key configuration items include:

## 3. Start Training

```bash
python train.py --config configs/my_model_1b.yaml
```

Or execute `build.sh` in the root directory to train all configurations.

## 4. Testing

```bash
python -m tests.test_generate_performance --config configs/TLinLLM_41M_1K.yaml
```

Or execute `test.sh` in the `tests` directory to run tests for all configurations.

## 5. Monitor Training Status

All metrics during the training process (such as loss, learning rate) will be recorded in the `results/runs/` directory. Use TensorBoard to view them:

```bash
tensorboard --logdir results/runs
```

Then open `http://localhost:6006` in your browser to see the real-time loss curve.


# Improvements Needed

## 1. Use MoE (Mixture of Experts)
## 2. Verify emergent abilities with larger parameters.
## 3. Verify capabilities like "Needle in a Haystack".

# More

See the `doc` directory.
