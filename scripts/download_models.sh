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
# ModelScope can close large transfers on hosted macOS runners. Downloads use
# resumable partial files, bounded retry/backoff, and lower group concurrency.
MODEL_DOWNLOAD_RETRIES="${MODEL_DOWNLOAD_RETRIES:-6}"
MODEL_DOWNLOAD_CONNECT_TIMEOUT="${MODEL_DOWNLOAD_CONNECT_TIMEOUT:-30}"
MAX_PARALLEL_MODEL_GROUPS="${MAX_PARALLEL_MODEL_GROUPS:-3}"

if ! [[ "$MODEL_DOWNLOAD_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MODEL_DOWNLOAD_RETRIES must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MODEL_DOWNLOAD_CONNECT_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MODEL_DOWNLOAD_CONNECT_TIMEOUT must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MAX_PARALLEL_MODEL_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_MODEL_GROUPS must be a positive integer" >&2
    exit 2
fi

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
    local partial="$dest.part"
    local attempt=1

    mkdir -p "$dest_dir"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ Already exists: $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    # Never write a transfer directly to its final path. Failed partial content
    # must not be mistaken for a completed model by later script invocations.
    while [ "$attempt" -le "$MODEL_DOWNLOAD_RETRIES" ]; do
        echo "Downloading $filename from ModelScope (attempt $attempt/$MODEL_DOWNLOAD_RETRIES)..."

        local resume_args=()
        if [ -s "$partial" ]; then
            resume_args=(--continue-at -)
            echo "  Resuming partial transfer ($(du -h "$partial" | cut -f1))"
        fi

        if curl --fail --location --silent --show-error \
            --connect-timeout "$MODEL_DOWNLOAD_CONNECT_TIMEOUT" \
            "${resume_args[@]}" \
            "$MODELSCOPE_BASE$ms_path" \
            --output "$partial"; then
            mv "$partial" "$dest"
            echo "✓ Downloaded: $filename ($(du -h "$dest" | cut -f1))"
            return 0
        fi

        if [ "$attempt" -lt "$MODEL_DOWNLOAD_RETRIES" ]; then
            # 2, 4, 6, 8, 10 second bounded backoff. Jitter prevents all
            # concurrent model groups reconnecting together.
            local delay=$((attempt * 2 + RANDOM % 3))
            echo "  Transfer failed; retrying in ${delay}s..." >&2
            sleep "$delay"
        fi
        attempt=$((attempt + 1))
    done

    echo "❌ Failed to download $filename after $MODEL_DOWNLOAD_RETRIES attempts (partial retained at: $partial)" >&2
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
    tier_failed=0

    # Each group launches its own model-file downloads. Limit active groups so
    # `--all` cannot create dozens of simultaneous large ModelScope transfers.
    start_model_group() {
        "$@" &
        tier_pids+=("$!")
        if [ "${#tier_pids[@]}" -ge "$MAX_PARALLEL_MODEL_GROUPS" ]; then
            if ! wait "${tier_pids[0]}"; then
                tier_failed=1
            fi
            tier_pids=("${tier_pids[@]:1}")
        fi
    }

    start_model_group download_ppocrv6 "tiny"   "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_tiny}"
    start_model_group download_ppocrv6 "small"  "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_small}"
    start_model_group download_ppocrv6 "medium" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6_medium}"
    start_model_group download_ppocrv5 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_mobile}"
    start_model_group download_ppocrv5 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_server}"
    start_model_group download_ppocrv4 "mobile" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_mobile}"
    start_model_group download_ppocrv4 "server" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_server}"

    for lang in "${V5_LANGS[@]}"; do
        start_model_group download_ppocrv5_lang "$lang" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_$lang}"
    done
    for lang in "${V4_LANGS[@]}"; do
        start_model_group download_ppocrv4_lang "$lang" "${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_$lang}"
    done

    if ! wait_jobs "${tier_pids[@]}"; then
        tier_failed=1
    fi
    if [ "$tier_failed" -ne 0 ]; then
        echo "❌ One or more model groups failed" >&2
        exit 1
    fi
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
