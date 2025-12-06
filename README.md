# Code-R1: Reproducing R1 for Code with Reliable Rewards (Multi-Node)

This repository includes implementations to reproduce the R1 pipeline for code generation:

* **Result:** 2K Code-R1 samples + `Qwen2.5-7B-Instruct-1M` beats `Qwen2.5-Coder-7B-Instruct` (even better w/ 12K samples).
* **Finding:** Quality of rewards matters. False positives in datasets and execution confuse the model.
* **Implementation:** A reliable, scalable, and sandboxed pipeline to minimize reward false positives in datasets and execution.

More results and findings to come...

## Setup

### Environment

```bash
# For training
pip install -e .
pip install vllm==0.6.3
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib gpustat # utility
```

> [!IMPORTANT]
>
> This branch is optimized for **multi-node training** (tested on 2×8 H100). For single-node setup, use `main` branch.

### Multi-Node Setup

For multi-node training, you need to:

1. Ensure SSH passwordless login between nodes
2. Sync the codebase to all nodes
3. Install dependencies on all nodes

```bash
# Sync to remote node
rsync -avz /home/ubuntu/code-r1/ <remote_ip>:/home/ubuntu/code-r1/

# Install on remote node
ssh <remote_ip> "cd /home/ubuntu/code-r1 && pip install -e . && pip install flash-attn --no-build-isolation"
```

Create hostfile at `~/hostfile.txt`:

```bash
<head_node_ip> slots=8
<worker_node_ip> slots=8
```

### Sandboxing

I tried multiple ways for sandboxing including calling code execution servers, running dockerized Python, calling paid services, etc.
`firejail` is the approach I found to meet all the three:

1. Reliability -- False positive comes when "autograders" have concurrency issue (timeouts), violating OS limits (`OSError`), etc.
2. Scalability -- e.g., dockerized Python run is generally reliable but too slow (e.g., 20 samples/s on 192 cores).
3. Security -- ... otherwise the school IT will email you and stop your server...

```bash
sudo add-apt-repository ppa:deki/firejail
sudo apt-get update
sudo apt-get install firejail firejail-profiles
```

### Datasets

The current version has 12K RL samples (prompt + tests) at [🤗 ganler/code-r1-12k](https://huggingface.co/datasets/ganler/code-r1-12k):

* [2K LeetCode data](https://github.com/newfacade/LeetCodeDataset) where the tests are generally reliable
* 10K verified data filtered from 26K [TACO](https://huggingface.co/datasets/BAAI/TACO) data.

In general, it's suggesgted to test data & sandbox on every dataset & environment before training code RL.
Directly using noisy data and mismatched envornments can lead to reward false positives, confusing the model.
These noise could come from (i) wrong tests, (ii) unsolvable prompts (e.g., images tags), and (iii) execution environment mismatch.

To produce locally validated RL data:

```bash
python examples/data_preprocess/coder1.py
```

## Run!

### Single-Node

```bash
bash main_grpo.sh
```

### Multi-Node

For multi-node training (e.g., 2×8 H100):

```bash
# 1. Verify InfiniBand connectivity (optional but recommended)
bash main_grpo_multinode.sh test

# 2. Start training
bash main_grpo_multinode.sh run
```

The multi-node script handles Ray cluster setup and NCCL configuration automatically.

> [!NOTE]
>
> Key things to configure in `main_grpo_multinode.sh`:
> * `HEAD_NODE_IP` and `WORKER_NODE_IP`: Your node IPs
> * `GRAD_ACC_STEPS`: Reduced from 4 to 2 (doubled GPUs, same global batch size)
>
> For NCCL issues across nodes, check `verl/trainer/main_ppo.py` where we set `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE=0`, etc.

For detailed multi-node debugging notes, see [multi_node.md](multi_node.md).

## Code-R1 Zero based on 7B models

We trained two models based on Qwen2.5-7B-Instruct-1M by pure R1 Zero:
* [🤗 CodeR1-Zero-Qwen2.5-7B-12k-832](https://huggingface.co/ganler/CodeR1-Zero-Qwen2.5-7B-12k-832): using 12K RL samples trained in 832 steps ([training logs](https://api.wandb.ai/links/llm4code/y13vs8d9)).
* [🤗 CodeR1-Zero-Qwen2.5-7B-LC2k-1088](https://huggingface.co/ganler/CodeR1-Zero-Qwen2.5-7B-LC2k-1088): using 2K RL samples from LeetCode,  trained in 1088 steps ([training logs](https://api.wandb.ai/links/llm4code/k8q6zu51)).

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-7B-Instruct-1M                         |     24.0      |     80.5       |    66.7     |   57.1      |
| + Code-R1-Zero (2k  - 1088s GRPO)              |     28.6      |     84.8       |    70.1     |   61.2      |
| + Code-R1-Zero (12k -  832s GRPO)              |     29.7      |     83.5       |    74.3     | 🌟**62.5**  |

* 2K leetcode training samples can already show promising results without any additional SFT or distillation.
* Adding it to 12K data (10K more verified data from TACO) can further improve the performance.

Some existing models:

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-Coder-7B-Instruct                      |     31.1      |     82.3       |    69.6     |  61.0       |
| Eurus-2-7B-PRIME                               |     23.8      |     65.9       |    29.9     |  39.9       |
| Sky-T1-7B                                      |     21.3      |     54.3       |    50.8     |  42.1       |

* Qwen2.5-Coder-7B-Instruct, despite released months back, is still very performant as the best baseline, but we don't know where the improvement comes from.
* Eurus-2-7B-PRIME starts from Qwen2.5-Math-7B-Instruct and is RL only. Its training data includes (unfiltered) extensive coding datasets, including APPS, CodeContests, TACO, and Codeforces. Code-R1-Zero outperforms it significantly despite using fewer data, likely because we use validated datasets and sandboxes.
* Sky-T1-7B uses a combination of RL and SFT/distillation steps. Its RL partially uses PRIME but its training data does not (seem to) include coding datasets.

## Citation

If you find this work helpful...

```bibtex
@article{code-r1,
  title={Code-R1: Reproducing R1 for Code with Reliable Rewards},
  author={Liu, Jiawei and Zhang, Lingming},
  howpublished={\url{https://github.com/ganler/code-r1}},
  year={2025}
}
```

## Acknowledgements

* [Verl](https://github.com/volcengine/verl)
* [Logic-RL](https://github.com/Unakar/Logic-RL)

## License

Apache-2.0. See [LICENSE.code-r1](LICENSE.code-r1) for more details.
