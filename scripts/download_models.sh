#!/bin/bash
# Script to download OCR models directly from Hugging Face Hub (byrizki/rusto-models)
# Source: https://huggingface.co/byrizki/rusto-models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_TYPE="ppocrv6"
TIER="tiny"
LANG=""
OUTPUT_DIR=""
DOWNLOAD_ALL=false
PREPARE_CI=false
CONVERT_RTEN=false
KEEP_ONNX=true
WITH_CLS=false
WITH_ORIENT=false
WITH_UNWARP=false
WITH_LAYOUT=false
WITH_MODULES=false
WITH_INDRA=false

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) DOWNLOAD_ALL=true ;;
        --ci|--test-fixtures|--benchmark) PREPARE_CI=true ;;
        --convert-rten) CONVERT_RTEN=true ;; # Kept for backward-compatibility, no-op since .rten is fetched directly
        --keep-onnx) KEEP_ONNX=true ;;
        --with-cls) WITH_CLS=true ;;
        --with-orient) WITH_ORIENT=true ;;
        --with-unwarp) WITH_UNWARP=true ;;
        --with-layout) WITH_LAYOUT=true ;;
        --with-modules) WITH_MODULES=true ;;
        --with-indra|--indra) WITH_INDRA=true ;;
        --model) MODEL_TYPE="$2"; shift ;;
        --tier) TIER="$2"; shift ;;
        --lang) LANG="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        *) OUTPUT_DIR="$1" ;;
    esac
    shift
done

HF_BASE="${HF_BASE:-https://huggingface.co/byrizki/rusto-models/resolve/main}"
MODEL_DOWNLOAD_RETRIES="${MODEL_DOWNLOAD_RETRIES:-5}"
MODEL_DOWNLOAD_CONNECT_TIMEOUT="${MODEL_DOWNLOAD_CONNECT_TIMEOUT:-20}"
MAX_PARALLEL_DOWNLOADS="${MAX_PARALLEL_DOWNLOADS:-6}"

# Wait for a list of PIDs; returns 1 if any failed
wait_jobs() {
    local failed=0
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    return $failed
}

download_file_to() {
    local dest_dir="$1"
    local filename="$2"
    local hf_rel_path="$3"
    local dest="$dest_dir/$filename"
    local partial="$dest.part"
    local attempt=1

    mkdir -p "$dest_dir"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ Already exists: $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    local url=""
    if [[ "$hf_rel_path" =~ ^https?:// ]]; then
        url="$hf_rel_path"
    else
        url="$HF_BASE/$hf_rel_path"
    fi

    while [ "$attempt" -le "$MODEL_DOWNLOAD_RETRIES" ]; do
        echo "Downloading $filename from Hugging Face (attempt $attempt/$MODEL_DOWNLOAD_RETRIES)..."

        local resume_args=()
        if [ -s "$partial" ]; then
            resume_args=(--continue-at -)
        fi

        if curl --fail --location --silent --show-error \
            --connect-timeout "$MODEL_DOWNLOAD_CONNECT_TIMEOUT" \
            "${resume_args[@]}" \
            "$url" \
            --output "$partial"; then
            mv "$partial" "$dest"
            echo "✓ Downloaded: $filename ($(du -h "$dest" | cut -f1))"
            return 0
        fi

        if [ "$attempt" -lt "$MODEL_DOWNLOAD_RETRIES" ]; then
            local delay=$((attempt * 2))
            echo "  Transfer failed; retrying in ${delay}s..." >&2
            sleep "$delay"
        fi
        attempt=$((attempt + 1))
    done

    echo "❌ Failed to download $filename after $MODEL_DOWNLOAD_RETRIES attempts" >&2
    return 1
}

download_ppocrv6() {
    local tier="${1:-tiny}"
    local dir="$2"
    echo "=== Downloading PP-OCRv6 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()
    local folder="ppocrv6-$tier"

    for ext in mnn onnx rten; do
        download_file_to "$dir" "det.$ext" "$folder/det.$ext" &
        pids+=($!)
        download_file_to "$dir" "rec.$ext" "$folder/rec.$ext" &
        pids+=($!)
    done

    download_file_to "$dir" "dict.txt" "$folder/dict.txt" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_ppocrv5() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv5 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()
    local folder="ppocrv5-$tier"

    for ext in onnx rten; do
        download_file_to "$dir" "det.$ext" "$folder/det.$ext" &
        pids+=($!)
        download_file_to "$dir" "rec.$ext" "$folder/rec.$ext" &
        pids+=($!)
    done

    download_file_to "$dir" "dict.txt" "$folder/dict.txt" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_ppocrv4() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv4 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()
    local folder="ppocrv4-$tier"

    for ext in onnx rten; do
        download_file_to "$dir" "det.$ext" "$folder/det.$ext" &
        pids+=($!)
        download_file_to "$dir" "rec.$ext" "$folder/rec.$ext" &
        pids+=($!)
    done

    download_file_to "$dir" "dict.txt" "$folder/dict.txt" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_modules() {
    local dir="$1"
    echo "=== Downloading Pipeline Modules (cls, orient, uvdoc, layout) ==="
    local pids=()

    # Classification
    download_file_to "$dir" "cls.mnn" "modules/cls.mnn" &
    pids+=($!)
    download_file_to "$dir" "cls.onnx" "modules/cls.onnx" &
    pids+=($!)
    download_file_to "$dir" "cls.rten" "modules/cls.rten" &
    pids+=($!)

    # Orientation
    download_file_to "$dir" "orient.mnn" "modules/orient.mnn" &
    pids+=($!)
    download_file_to "$dir" "orient.onnx" "modules/orient.onnx" &
    pids+=($!)
    download_file_to "$dir" "orient.rten" "modules/orient.rten" &
    pids+=($!)

    # UVDoc Unwarp
    download_file_to "$dir" "uvdoc.mnn" "modules/uvdoc.mnn" &
    pids+=($!)
    download_file_to "$dir" "uvdoc.onnx" "modules/uvdoc.onnx" &
    pids+=($!)
    download_file_to "$dir" "uvdoc.rten" "modules/uvdoc.rten" &
    pids+=($!)

    # Layout CDLA & PubLayNet
    download_file_to "$dir" "layout_cdla.mnn" "modules/layout_cdla.mnn" &
    pids+=($!)
    download_file_to "$dir" "layout_cdla.onnx" "modules/layout_cdla.onnx" &
    pids+=($!)
    download_file_to "$dir" "layout_cdla.rten" "modules/layout_cdla.rten" &
    pids+=($!)
    download_file_to "$dir" "layout_publaynet.mnn" "modules/layout_publaynet.mnn" &
    pids+=($!)
    download_file_to "$dir" "layout_publaynet.onnx" "modules/layout_publaynet.onnx" &
    pids+=($!)
    download_file_to "$dir" "layout_publaynet.rten" "modules/layout_publaynet.rten" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_indra() {
    local dir="$1"
    echo "=== Downloading Indra OCR Models (Hugging Face) ==="
    mkdir -p "$dir"
    local pids=()

    # Core Detection & Recognition
    for ext in mnn onnx rten; do
        download_file_to "$dir" "det.$ext" "indraocr/det.$ext" &
        pids+=($!)
        download_file_to "$dir" "rec.$ext" "indraocr/rec.$ext" &
        pids+=($!)
    done

    download_file_to "$dir" "meta.json" "indraocr/meta.json" &
    pids+=($!)

    # Reading Order & Orientation
    for ext in mnn onnx rten; do
        download_file_to "$dir" "order.$ext" "indraocr/order.$ext" &
        pids+=($!)
        download_file_to "$dir" "orient.$ext" "indraocr/orient.$ext" &
        pids+=($!)
    done

    # Multi-Script Recognition
    for variant in rec_cjk rec_rtl rec_world script mathcls; do
        for ext in onnx rten; do
            download_file_to "$dir" "${variant}.$ext" "indraocr/${variant}.$ext" &
            pids+=($!)
        done
    done

    # LaTeX Math Recognition
    for part in math_enc math_dec_init math_dec_step; do
        for ext in onnx rten; do
            download_file_to "$dir" "${part}.$ext" "indraocr/${part}.$ext" &
            pids+=($!)
        done
    done
    download_file_to "$dir" "math_meta.json" "indraocr/math_meta.json" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_ci_fixtures() {
    echo "=== Preparing All Test Model Fixtures for CI / Benchmark ==="
    local ppocr_dir="$REPO_ROOT/models/PPOCR_v6_tiny"
    local ppocr_small_dir="$REPO_ROOT/models/PPOCR_v6_small"
    local indra_dir="$REPO_ROOT/models/indra"
    local modules_dir="$REPO_ROOT/models"

    download_ppocrv6 "tiny" "$ppocr_dir"
    download_ppocrv6 "small" "$ppocr_small_dir"
    download_modules "$ppocr_dir"
    download_indra "$indra_dir"

    # Also symlink / copy module fixtures to models/ if referenced without subpath
    for mod in cls orient uvdoc layout_cdla layout_publaynet; do
        for ext in mnn onnx rten; do
            if [ -f "$ppocr_dir/$mod.$ext" ] && [ ! -f "$modules_dir/$mod.$ext" ]; then
                cp -p "$ppocr_dir/$mod.$ext" "$modules_dir/$mod.$ext" 2>/dev/null || true
            fi
        done
    done
}

if [ "$PREPARE_CI" = true ]; then
    download_ci_fixtures
elif [ "$DOWNLOAD_ALL" = true ]; then
    download_ppocrv6 "tiny" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_tiny}"
    download_ppocrv6 "small" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_small}"
    download_ppocrv6 "medium" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_medium}"
    download_ppocrv5 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_mobile}"
    download_ppocrv4 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_mobile}"
    download_modules "${OUTPUT_DIR:-$REPO_ROOT/models}"
    download_indra "${OUTPUT_DIR:-$REPO_ROOT/models/indra}"
else
    TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_$TIER}"
    case "$MODEL_TYPE" in
        ppocrv6)
            download_ppocrv6 "$TIER" "$TARGET_DIR"
            ;;
        ppocrv5)
            TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_$TIER}"
            download_ppocrv5 "$TIER" "$TARGET_DIR"
            ;;
        ppocrv4)
            TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_$TIER}"
            download_ppocrv4 "$TIER" "$TARGET_DIR"
            ;;
        indra)
            TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/indra}"
            download_indra "$TARGET_DIR"
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac

    if [ "$WITH_MODULES" = true ] || [ "$WITH_CLS" = true ] || [ "$WITH_ORIENT" = true ] || [ "$WITH_UNWARP" = true ] || [ "$WITH_LAYOUT" = true ]; then
        download_modules "$TARGET_DIR"
    fi

    if [ "$WITH_INDRA" = true ]; then
        download_indra "${OUTPUT_DIR:-$REPO_ROOT/models/indra}"
    fi
fi

echo ""
echo "✅ Model download from Hugging Face complete."
