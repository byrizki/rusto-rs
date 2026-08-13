#!/bin/bash
# Script to download OCR models directly from ModelScope
# Source: https://www.modelscope.cn/models/RapidAI/RapidOCR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_TYPE="ppocrv6"
TIER="tiny"
LANG=""
OUTPUT_DIR=""
DOWNLOAD_ALL=false

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) DOWNLOAD_ALL=true ;;
        --model) MODEL_TYPE="$2"; shift ;;
        --tier) TIER="$2"; shift ;;
        --lang) LANG="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        *) OUTPUT_DIR="$1" ;;
    esac
    shift
done

MODELSCOPE_BASE="https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath="

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

    echo "❌ Failed to download $filename (dest: $dest)"
    return 1
}

download_ppocrv6() {
    local tier="$1"
    local dir="$2"
    echo "=== Downloading PP-OCRv6 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()

    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_${tier}.mnn" &
    pids+=($!)

    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_${tier}.mnn" &
    pids+=($!)

    if [ "$tier" = "tiny" ]; then
        download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny%2Fppocrv6_tiny_dict.txt" &
    else
        download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_small%2Fppocrv6_dict.txt" &
    fi
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_ppocrv5() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv5 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()

    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv5%2Fdet%2Fch_PP-OCRv5_det_${tier}.mnn" &
    pids+=($!)

    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}.mnn" &
    pids+=($!)

    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}%2Fppocrv5_dict.txt" &
    pids+=($!)

    # English rec model is only available for the mobile tier
    if [ "$tier" = "mobile" ]; then
        download_file_to "$dir" "rec_en.mnn" "mnn%2FPP-OCRv5%2Frec%2Fen_PP-OCRv5_rec_${tier}.mnn" &
        pids+=($!)

        download_file_to "$dir" "dict_en.txt" "paddle%2FPP-OCRv5%2Frec%2Fen_PP-OCRv5_rec_${tier}%2Fppocrv5_en_dict.txt" &
        pids+=($!)
    fi

    wait_jobs "${pids[@]}"
}

download_ppocrv5_lang() {
    local lang="$1"
    local dir="$2"
    echo "=== Downloading PP-OCRv5 ($lang) model ==="
    echo "Destination: $dir"

    local pids=()

    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv5%2Frec%2F${lang}_PP-OCRv5_rec_mobile.mnn" &
    pids+=($!)

    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv5%2Frec%2F${lang}_PP-OCRv5_rec_mobile%2Fppocrv5_${lang}_dict.txt" &
    pids+=($!)

    wait_jobs "${pids[@]}"
}

download_ppocrv4() {
    local tier="${1:-mobile}"
    local dir="$2"
    echo "=== Downloading PP-OCRv4 ($tier tier) models ==="
    echo "Destination: $dir"

    local pids=()

    download_file_to "$dir" "det.mnn" "mnn%2FPP-OCRv4%2Fdet%2Fch_PP-OCRv4_det_${tier}.mnn" &
    pids+=($!)

    download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_${tier}.mnn" &
    pids+=($!)

    # dict.txt only exists in the mobile rec directory on ModelScope; shared by both tiers
    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_mobile%2Fppocr_keys_v1.txt" &
    pids+=($!)

    # English rec model and cls are only available for the mobile tier
    if [ "$tier" = "mobile" ]; then
        download_file_to "$dir" "rec_en.mnn" "mnn%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}.mnn" &
        pids+=($!)

        download_file_to "$dir" "dict_en.txt" "paddle%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}%2Fen_dict.txt" &
        pids+=($!)

        download_file_to "$dir" "cls.mnn" "mnn%2FPP-OCRv4%2Fcls%2Fch_ppocr_mobile_v2.0_cls_${tier}.mnn" &
        pids+=($!)
    fi

    wait_jobs "${pids[@]}"
}

download_ppocrv4_lang() {
    local lang="$1"
    local dir="$2"
    echo "=== Downloading PP-OCRv4 ($lang) model ==="
    echo "Destination: $dir"

    local pids=()

    case "$lang" in
        japan)
            download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2Fjapan_PP-OCRv4_rec_mobile.mnn" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fjapan_PP-OCRv4_rec_mobile%2Fjapan_dict.txt" &
            pids+=($!)
            ;;
        chinese_cht|chinese-cht)
            download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2Fchinese_cht_PP-OCRv3_rec_mobile.mnn" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fchinese_cht_PP-OCRv3_rec_mobile%2Fchinese_cht_dict.txt" &
            pids+=($!)
            ;;
        kannada|ka)
            download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2Fka_PP-OCRv4_rec_mobile.mnn" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fkannada_PP-OCRv4_rec_mobile%2Fka_dict.txt" &
            pids+=($!)
            ;;
        *)
            download_file_to "$dir" "rec.mnn" "mnn%2FPP-OCRv4%2Frec%2F${lang}_PP-OCRv4_rec_mobile.mnn" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2F${lang}_PP-OCRv4_rec_mobile%2F${lang}_dict.txt" &
            pids+=($!)
            ;;
    esac

    wait_jobs "${pids[@]}"
}

V5_LANGS=("arabic" "cyrillic" "devanagari" "el" "eslav" "korean" "latin" "ta" "te" "th")
V4_LANGS=("japan" "chinese_cht" "kannada")

if [ "$DOWNLOAD_ALL" = true ]; then
    tier_pids=()

    download_ppocrv6 "tiny"   "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_tiny}"   & tier_pids+=($!)
    download_ppocrv6 "small"  "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_small}"  & tier_pids+=($!)
    download_ppocrv6 "medium" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_medium}" & tier_pids+=($!)
    download_ppocrv5 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_mobile}" & tier_pids+=($!)
    download_ppocrv5 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_server}" & tier_pids+=($!)
    download_ppocrv4 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_mobile}" & tier_pids+=($!)
    download_ppocrv4 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_server}" & tier_pids+=($!)

    for lang in "${V5_LANGS[@]}"; do
        download_ppocrv5_lang "$lang" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_$lang}" & tier_pids+=($!)
    done
    for lang in "${V4_LANGS[@]}"; do
        download_ppocrv4_lang "$lang" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_$lang}" & tier_pids+=($!)
    done

    wait_jobs "${tier_pids[@]}"
else
    case "$MODEL_TYPE" in
        ppocrv6)
            download_ppocrv6 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6}"
            ;;
        ppocrv5)
            if [ -n "$LANG" ]; then
                download_ppocrv5_lang "$LANG" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_$LANG}"
            else
                download_ppocrv5 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5}"
            fi
            ;;
        ppocrv4)
            if [ -n "$LANG" ]; then
                download_ppocrv4_lang "$LANG" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_$LANG}"
            else
                download_ppocrv4 "$TIER" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4}"
            fi
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ Download complete"
