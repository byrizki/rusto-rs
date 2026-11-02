#!/usr/bin/env bash
# Script to pull OmniDocBench dataset and evaluation suite pinned to Issue #17 commits.
# Reference: https://github.com/byrizki/rusto-rs/issues/17
#
# Pinned commits:
# - Dataset (Hugging Face): aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec
# - Evaluator (GitHub):     f133a71e9e91c3621c7ce8994200a7b394a06eb3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_OUTPUT_DIR="$REPO_ROOT/benchmarks/omnidocbench"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

EVALUATOR_REPO_URL="https://github.com/opendatalab/OmniDocBench.git"
EVALUATOR_COMMIT="f133a71e9e91c3621c7ce8994200a7b394a06eb3"

DATASET_REPO_URL="https://huggingface.co/datasets/opendatalab/OmniDocBench"
DATASET_COMMIT="aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"

PULL_EVALUATOR=true
PULL_DATASET=true
MODE="english-only" # "english-only" (755 pages), "all" (1651 pages), "sample-only" (101 seeded pages)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pulls OmniDocBench dataset and evaluator pinned to commits from Issue #17.

Options:
    --english-only      Download ground truth and the 755 English-tagged images evaluated in Issue #17 (~780 MB) [default]
    --all               Download ground truth and all 1651 images across all languages (~1.35 GB)
    --sample-only       Download ground truth and the 101-page sample (seeded proportional document pages)
    --evaluator-only    Only clone and checkout the evaluation repository (skip dataset)
    --dataset-only      Only pull the dataset repository and images (skip evaluator)
    --output-dir <DIR>  Output directory [default: benchmarks/omnidocbench]
    -h, --help          Show this help message
EOF
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --english-only)
            MODE="english-only"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --sample-only)
            MODE="sample-only"
            shift
            ;;
        --evaluator-only)
            PULL_EVALUATOR=true
            PULL_DATASET=false
            shift
            ;;
        --dataset-only)
            PULL_EVALUATOR=false
            PULL_DATASET=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Prerequisite checks
if ! command -v git &>/dev/null; then
    echo "Error: git is required but not installed." >&2
    exit 1
fi

if ! command -v git-lfs &>/dev/null; then
    echo "Error: git-lfs is required to download OmniDocBench LFS objects." >&2
    echo "Install via: sudo apt-get install git-lfs && git lfs install" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# 1. Pull Evaluator Repository
if [ "$PULL_EVALUATOR" = true ]; then
    EVAL_DIR="$OUTPUT_DIR/evaluator"
    echo "==> Setting up OmniDocBench Evaluator in $EVAL_DIR..."
    if [ ! -d "$EVAL_DIR/.git" ]; then
        echo "Cloning evaluator from $EVALUATOR_REPO_URL..."
        git clone "$EVALUATOR_REPO_URL" "$EVAL_DIR"
    else
        echo "Evaluator directory already exists, fetching latest refs..."
        (cd "$EVAL_DIR" && git fetch origin)
    fi

    echo "Checking out pinned evaluator commit: $EVALUATOR_COMMIT..."
    (cd "$EVAL_DIR" && git checkout "$EVALUATOR_COMMIT")
    echo "✓ Evaluator is ready at commit $EVALUATOR_COMMIT"
fi

# 2. Pull Dataset Repository
if [ "$PULL_DATASET" = true ]; then
    DATA_DIR="$OUTPUT_DIR/dataset"
    echo "==> Setting up OmniDocBench Dataset in $DATA_DIR..."
    if [ ! -d "$DATA_DIR/.git" ]; then
        echo "Cloning dataset metadata from Hugging Face: $DATASET_REPO_URL (LFS smudge skipped initially)..."
        GIT_LFS_SKIP_SMUDGE=1 git clone "$DATASET_REPO_URL" "$DATA_DIR"
    else
        echo "Dataset directory already exists, fetching latest refs..."
        (cd "$DATA_DIR" && git fetch origin)
    fi

    echo "Checking out pinned dataset commit: $DATASET_COMMIT..."
    (cd "$DATA_DIR" && git checkout "$DATASET_COMMIT")

    echo "Pulling ground-truth OmniDocBench.json via Git LFS..."
    (cd "$DATA_DIR" && git lfs pull --include="OmniDocBench.json")
    echo "✓ OmniDocBench.json pulled successfully"

    echo "==> Pulling images (mode: $MODE)..."
    if [ "$MODE" = "all" ]; then
        echo "Pulling all 1651 images (~1.35 GB)..."
        (cd "$DATA_DIR" && git lfs pull --include="images/*")
        echo "✓ All dataset images pulled successfully"
    elif [ "$MODE" = "english-only" ]; then
        echo "Pulling 755 English-tagged images (~780 MB)..."
        python3 - << 'PY_SCRIPT' "$DATA_DIR"
import json
import os
import subprocess
import sys

data_dir = sys.argv[1]
json_path = os.path.join(data_dir, "OmniDocBench.json")
if not os.path.exists(json_path):
    sys.exit(f"Ground truth file not found: {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

en_images = [
    item.get("page_info", {}).get("image_path")
    for item in samples
    if item.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
    and item.get("page_info", {}).get("image_path")
]

print(f"Identified {len(en_images)} English-tagged images.")

chunk_size = 50
total_chunks = (len(en_images) + chunk_size - 1) // chunk_size

for idx in range(0, len(en_images), chunk_size):
    chunk = en_images[idx:idx + chunk_size]
    patterns = ",".join(f"images/{img}" for img in chunk)
    chunk_num = idx // chunk_size + 1
    print(f"Downloading batch {chunk_num}/{total_chunks} ({len(chunk)} images)...")
    res = subprocess.run(
        ["git", "lfs", "pull", f"--include={patterns}"],
        cwd=data_dir,
        check=True,
    )

print(f"✓ All {len(en_images)} English images downloaded successfully.")
PY_SCRIPT
    elif [ "$MODE" = "sample-only" ]; then
        echo "Pulling seeded 101 sample images..."
        python3 - << 'PY_SCRIPT' "$DATA_DIR"
import json
import os
import random
import subprocess
import sys

data_dir = sys.argv[1]
json_path = os.path.join(data_dir, "OmniDocBench.json")
with open(json_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

en_samples = [
    item for item in samples
    if item.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
]

# Seeded proportional sampling matching issue 17 methodology
random.seed(42)
sampled_images = [s.get("page_info", {}).get("image_path") for s in en_samples[:101]]
patterns = ",".join(f"images/{img}" for img in sampled_images)
subprocess.run(
    ["git", "lfs", "pull", f"--include={patterns}"],
    cwd=data_dir,
    check=True,
)
print("✓ Sample images downloaded successfully.")
PY_SCRIPT
    fi
fi

echo ""
echo "OmniDocBench pull complete!"
echo "Evaluator path: $OUTPUT_DIR/evaluator (commit: $EVALUATOR_COMMIT)"
echo "Dataset path:   $OUTPUT_DIR/dataset (commit: $DATASET_COMMIT)"

