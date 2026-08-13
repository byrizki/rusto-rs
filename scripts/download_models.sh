#!/bin/bash
# Script to download OCR models directly from ModelScope
# Source: https://www.modelscope.cn/models/RapidAI/RapidOCR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_TYPE="ppocrv6"
TIER="tiny"
OUTPUT_DIR=""
DOWNLOAD_ALL=false

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) DOWNLOAD_ALL=true ;;
        --model) MODEL_TYPE="$2"; shift ;;
        --tier) TIER="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        *) OUTPUT_DIR="$1" ;;
    esac
    shift
done

MODELSCOPE_BASE="https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath="

download_file_to() {
    local dest_dir="$1"
    local filename="$2"
    local ms_path="$3"
    local dest="$dest_dir/$filename"

    mkdir -p "$dest_dir"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ Already exists: $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    echo "Downloading $filename from ModelScope..."
    if curl -sSL -f "$MODELSCOPE_BASE$ms_path" -o "$dest"; then
        echo "✓ Downloaded: $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    echo "❌ Failed to download $filename"
    return 1
}

download_ppocrv6() {
    local tier="$1"
    local dir="$2"
    echo "=== Downloading PP-OCRv6 ($tier tier) models ==="
    echo "Destination: $dir"
    
    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_${tier}.mnn"
    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_${tier}.mnn"
    
    if [ "$tier" = "tiny" ]; then
        download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny%2Fppocrv6_tiny_dict.txt"
    else
        download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_small%2Fppocrv6_dict.txt"
    fi
}

download_ppocrv5() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv5 ($tier tier) models ==="
    echo "Destination: $dir"
    
    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv5%2Fdet%2Fch_PP-OCRv5_det_${tier}.mnn"
    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}.mnn"
    download_file_to "$dir" "rec_en.mnn" "mnn%2FPP-OCRv5%2Frec%2Fen_PP-OCRv5_rec_${tier}.mnn"
    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}%2Fppocrv5_dict.txt"
    download_file_to "$dir" "dict_en.txt" "paddle%2FPP-OCRv5%2Frec%2Fen_PP-OCRv5_rec_${tier}%2Fppocrv5_en_dict.txt"
}

download_ppocrv4() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv4 ($tier tier) models ==="
    echo "Destination: $dir"
    
    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv4%2Fdet%2Fch_PP-OCRv4_det_${tier}.mnn"
    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_${tier}.mnn"
    download_file_to "$dir" "rec_en.mnn" "mnn%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}.mnn"
    download_file_to "$dir" "cls.mnn" "mnn%2FPP-OCRv4%2Fcls%2Fch_ppocr_mobile_v2.0_cls_${tier}.mnn"
    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_${tier}%2Fppocr_keys_v1.txt"
    download_file_to "$dir" "dict_en.txt" "paddle%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}%2Fen_dict.txt"
}

if [ "$DOWNLOAD_ALL" = true ]; then
    download_ppocrv6 "tiny" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_tiny}"
    download_ppocrv6 "small" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_small}"
    download_ppocrv6 "medium" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_medium}"
    download_ppocrv5 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_mobile}"
    download_ppocrv5 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_server}"
    download_ppocrv4 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_mobile}"
    download_ppocrv4 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_server}"
else
    case "$MODEL_TYPE" in
        ppocrv6)
            download_ppocrv6 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6}"
            ;;
        ppocrv5)
            download_ppocrv5 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5}"
            ;;
        ppocrv4)
            download_ppocrv4 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4}"
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ Download complete"
