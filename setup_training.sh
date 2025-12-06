#!/bin/bash
# Setup script for Code-R1 training on multi-node cluster
set -e

cd /home/ubuntu/code-r1

echo "=========================================="
echo "Setting up Code-R1 Training Environment"
echo "=========================================="

# ==============================================================================
# 1. Create directories
# ==============================================================================
echo "[1/5] Creating directories..."
mkdir -p data/code-r1-12k
mkdir -p models

# ==============================================================================
# 2. Download dataset from HuggingFace
# ==============================================================================
echo "[2/5] Downloading dataset from HuggingFace..."
python3 << 'EOF'
from datasets import load_dataset
import os

print("Loading ganler/code-r1-12k dataset...")
dataset = load_dataset("ganler/code-r1-12k")

output_dir = "data/code-r1-12k"
os.makedirs(output_dir, exist_ok=True)

print("Saving train split...")
dataset["train"].to_parquet(os.path.join(output_dir, "train.parquet"))

print("Saving test split...")
dataset["test"].to_parquet(os.path.join(output_dir, "test.parquet"))

print(f"Dataset saved to {output_dir}")
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
EOF

# ==============================================================================
# 3. Download model
# ==============================================================================
echo "[3/5] Downloading Qwen2.5-7B-Instruct-1M model..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

model_path = "models/Qwen2.5-7B-Instruct-1M"
if not os.path.exists(model_path) or not os.listdir(model_path):
    print("Downloading Qwen/Qwen2.5-7B-Instruct-1M...")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct-1M",
        local_dir=model_path,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {model_path}")
else:
    print(f"Model already exists at {model_path}")
EOF

# ==============================================================================
# 4. Setup wandb
# ==============================================================================
echo "[4/5] Setting up wandb..."
export WANDB_API_KEY="7b086d88f5a4dada0caedfc027e3eb69f166b941"
wandb login --relogin ${WANDB_API_KEY}

# ==============================================================================
# 5. Verify firejail installation
# ==============================================================================
echo "[5/5] Checking firejail installation..."
if command -v firejail &> /dev/null; then
    echo "firejail is installed: $(firejail --version | head -1)"
else
    echo "WARNING: firejail not installed. Installing..."
    sudo add-apt-repository -y ppa:deki/firejail
    sudo apt-get update
    sudo apt-get install -y firejail firejail-profiles
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  bash main_grpo_multinode.sh run"
echo ""
echo "Or to just setup Ray cluster:"
echo "  bash main_grpo_multinode.sh setup"

