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
CACHE_DIR="${OMNIDOCBENCH_CACHE_DIR:-}"

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
    --english-only, --english
                        Download ground truth and the 755 English-tagged images evaluated in Issue #17 (~780 MB) [default]
    --all, --all-languages
                        Download ground truth and all 1651 images across all languages (~1.35 GB)
    --sample-only, --sample
                        Download ground truth and the 101-page sample (seeded proportional document pages)
    --evaluator-only    Only clone and checkout the evaluation repository (skip dataset)
    --dataset-only      Only pull the dataset repository and images (skip evaluator)
    --cache-dir <DIR>   Specify persistent cache directory to restore/sync dataset
    --output-dir <DIR>  Output directory [default: benchmarks/omnidocbench]
    -h, --help          Show this help message
EOF
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --english-only|--english)
            MODE="english-only"
            shift
            ;;
        --all|--all-languages)
            MODE="all"
            shift
            ;;
        --sample-only|--sample)
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
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
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

# 1. Pull / Verify Evaluator Repository
if [ "$PULL_EVALUATOR" = true ]; then
    EVAL_DIR="$OUTPUT_DIR/evaluator"
    echo "==> Setting up OmniDocBench Evaluator in $EVAL_DIR..."
    if [ ! -d "$EVAL_DIR/.git" ]; then
        echo "Cloning evaluator from $EVALUATOR_REPO_URL..."
        git clone "$EVALUATOR_REPO_URL" "$EVAL_DIR"
        (cd "$EVAL_DIR" && git checkout "$EVALUATOR_COMMIT")
        echo "✓ Evaluator cloned and checked out at $EVALUATOR_COMMIT"
    else
        CURRENT_REV="$(cd "$EVAL_DIR" && git rev-parse HEAD 2>/dev/null || echo "")"
        if [ "$CURRENT_REV" = "$EVALUATOR_COMMIT" ]; then
            echo "✓ Evaluator already cached and up to date at commit $EVALUATOR_COMMIT"
        else
            echo "Fetching latest refs for evaluator..."
            (cd "$EVAL_DIR" && git fetch origin && git checkout "$EVALUATOR_COMMIT")
            echo "✓ Checked out evaluator commit $EVALUATOR_COMMIT"
        fi
    fi
fi

# 2. Pull / Verify Dataset Repository
if [ "$PULL_DATASET" = true ]; then
    DATA_DIR="$OUTPUT_DIR/dataset"
    echo "==> Setting up OmniDocBench Dataset in $DATA_DIR..."

    # If external CACHE_DIR provided and DATA_DIR not initialized, seed from CACHE_DIR
    if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR/dataset/.git" ] && [ ! -d "$DATA_DIR/.git" ]; then
        echo "Restoring dataset repository from cache: $CACHE_DIR/dataset..."
        mkdir -p "$DATA_DIR"
        cp -a "$CACHE_DIR/dataset/." "$DATA_DIR/"
    fi

    if [ ! -d "$DATA_DIR/.git" ]; then
        echo "Cloning dataset metadata from Hugging Face: $DATASET_REPO_URL (LFS smudge skipped initially)..."
        GIT_LFS_SKIP_SMUDGE=1 git clone "$DATASET_REPO_URL" "$DATA_DIR"
        (cd "$DATA_DIR" && git checkout "$DATASET_COMMIT")
    else
        CURRENT_DATASET_REV="$(cd "$DATA_DIR" && git rev-parse HEAD 2>/dev/null || echo "")"
        if [ "$CURRENT_DATASET_REV" = "$DATASET_COMMIT" ]; then
            echo "✓ Dataset repository already at pinned commit $DATASET_COMMIT"
        else
            echo "Fetching latest refs for dataset..."
            (cd "$DATA_DIR" && git fetch origin && git checkout "$DATASET_COMMIT")
        fi
    fi

    # Verify ground-truth OmniDocBench.json
    GT_FILE="$DATA_DIR/OmniDocBench.json"
    if [ -f "$GT_FILE" ] && [ "$(stat -c%s "$GT_FILE" 2>/dev/null || stat -f%z "$GT_FILE" 2>/dev/null || echo 0)" -gt 1000000 ]; then
        echo "✓ Ground-truth OmniDocBench.json already cached ($(du -h "$GT_FILE" | cut -f1))"
    else
        echo "Pulling ground-truth OmniDocBench.json via Git LFS..."
        (cd "$DATA_DIR" && git lfs pull --include="OmniDocBench.json")
        echo "✓ OmniDocBench.json pulled successfully"
    fi

    # Check external cache for images if available
    if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR/dataset/images" ]; then
        echo "Restoring cached images from $CACHE_DIR/dataset/images..."
        mkdir -p "$DATA_DIR/images"
        cp -rn "$CACHE_DIR/dataset/images"/* "$DATA_DIR/images/" 2>/dev/null || true
    fi

    echo "==> Verifying and pulling images (mode: $MODE)..."
    python3 - << 'PY_SCRIPT' "$DATA_DIR" "$MODE"
import json
import os
import random
import subprocess
import sys
import time

data_dir = sys.argv[1]
mode = sys.argv[2]
json_path = os.path.join(data_dir, "OmniDocBench.json")

if not os.path.exists(json_path):
    sys.exit(f"Ground truth file not found: {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

if mode in ("all", "all-languages"):
    target_images = [
        item.get("page_info", {}).get("image_path")
        for item in samples
        if item.get("page_info", {}).get("image_path")
    ]
elif mode in ("english-only", "english"):
    target_images = [
        item.get("page_info", {}).get("image_path")
        for item in samples
        if item.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
        and item.get("page_info", {}).get("image_path")
    ]
elif mode in ("sample-only", "sample"):
    en_samples = [
        item for item in samples
        if item.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
    ]
    random.seed(42)
    target_images = [s.get("page_info", {}).get("image_path") for s in en_samples[:101]]
else:
    sys.exit(f"Unknown mode: {mode}")

images_dir = os.path.join(data_dir, "images")
os.makedirs(images_dir, exist_ok=True)

def is_cached_image(img_name):
    if not img_name:
        return False
    p = os.path.join(images_dir, img_name)
    # LFS pointer stubs are ~130 bytes, real images are > 1000 bytes
    return os.path.isfile(p) and os.path.getsize(p) > 1000

already_cached = [img for img in target_images if is_cached_image(img)]
needed = [img for img in target_images if not is_cached_image(img)]

print(f"Dataset mode: '{mode}' ({len(target_images)} target images)")
print(f"✓ Cached on disk: {len(already_cached)}/{len(target_images)} images")

if not needed:
    print(f"✓ All {len(target_images)} images are already downloaded and cached. Nothing to pull.")
    sys.exit(0)

print(f"Downloading {len(needed)} missing images via Git LFS...")

chunk_size = 50
total_chunks = (len(needed) + chunk_size - 1) // chunk_size
max_retries = 3

for idx in range(0, len(needed), chunk_size):
    chunk = needed[idx:idx + chunk_size]
    # In Git LFS, --include treats comma as delimiter and brackets as globs.
    # Sanitize each filename so Git LFS matches the literal file.
    patterns = ",".join(
        f"images/{img.replace(',', '?').replace('[', '?').replace(']', '?')}"
        for img in chunk
    )
    chunk_num = idx // chunk_size + 1
    print(f"Downloading batch {chunk_num}/{total_chunks} ({len(chunk)} images)...")

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["git", "lfs", "pull", f"--include={patterns}"],
                cwd=data_dir,
                check=True,
            )
            break
        except subprocess.CalledProcessError as e:
            if attempt == max_retries:
                print(f"Error pulling batch {chunk_num} after {max_retries} attempts.")
                raise e
            print(f"Batch {chunk_num} attempt {attempt} failed, retrying in 3s...")
            time.sleep(3)

verified_count = sum(1 for img in target_images if is_cached_image(img))
print(f"✓ Download complete: {verified_count}/{len(target_images)} images verified on disk.")
PY_SCRIPT

    # Sync back to CACHE_DIR if set
    if [ -n "$CACHE_DIR" ]; then
        echo "Updating cache directory: $CACHE_DIR/dataset..."
        mkdir -p "$CACHE_DIR/dataset/images"
        cp -rn "$DATA_DIR/images"/* "$CACHE_DIR/dataset/images/" 2>/dev/null || true
        cp -n "$DATA_DIR/OmniDocBench.json" "$CACHE_DIR/dataset/OmniDocBench.json" 2>/dev/null || true
    fi
fi

echo ""
echo "OmniDocBench pull complete!"
echo "Evaluator path: $OUTPUT_DIR/evaluator (commit: $EVALUATOR_COMMIT)"
echo "Dataset path:   $OUTPUT_DIR/dataset (commit: $DATASET_COMMIT)"
