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
CONVERT_RTEN=false
KEEP_ONNX=false
WITH_CLS=false
WITH_ORIENT=false
WITH_UNWARP=false
WITH_LAYOUT=false

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) DOWNLOAD_ALL=true ;;
        --convert-rten) CONVERT_RTEN=true ;;
        --keep-onnx) KEEP_ONNX=true ;;
        --with-cls) WITH_CLS=true ;;
        --with-orient) WITH_ORIENT=true ;;
        --with-unwarp) WITH_UNWARP=true ;;
        --with-layout) WITH_LAYOUT=true ;;
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

        local url=""
        if [[ "$ms_path" =~ ^https?:// ]]; then
            url="$ms_path"
        else
            url="$MODELSCOPE_BASE$ms_path"
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

    download_file_to "$dir" "det.onnx" "onnx%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_${tier}.onnx" &
    pids+=($!)

    download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_${tier}.onnx" &
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

    download_file_to "$dir" "det.onnx" "onnx%2FPP-OCRv5%2Fdet%2Fch_PP-OCRv5_det_${tier}.onnx" &
    pids+=($!)

    download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}.onnx" &
    pids+=($!)

    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv5%2Frec%2Fch_PP-OCRv5_rec_${tier}%2Fppocrv5_dict.txt" &
    pids+=($!)

    # English rec model is only available for the mobile tier
    if [ "$tier" = "mobile" ]; then
        download_file_to "$dir" "rec_en.onnx" "onnx%2FPP-OCRv5%2Frec%2Fen_PP-OCRv5_rec_${tier}.onnx" &
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

    download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv5%2Frec%2F${lang}_PP-OCRv5_rec_mobile.onnx" &
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

    download_file_to "$dir" "det.onnx" "onnx%2FPP-OCRv4%2Fdet%2Fch_PP-OCRv4_det_${tier}.onnx" &
    pids+=($!)

    download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_${tier}.onnx" &
    pids+=($!)

    # dict.txt only exists in the mobile rec directory on ModelScope; shared by both tiers
    download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fch_PP-OCRv4_rec_mobile%2Fppocr_keys_v1.txt" &
    pids+=($!)

    # English rec model and cls are only available for the mobile tier
    if [ "$tier" = "mobile" ]; then
        download_file_to "$dir" "rec_en.onnx" "onnx%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}.onnx" &
        pids+=($!)

        download_file_to "$dir" "dict_en.txt" "paddle%2FPP-OCRv4%2Frec%2Fen_PP-OCRv4_rec_${tier}%2Fen_dict.txt" &
        pids+=($!)

        download_file_to "$dir" "cls.onnx" "onnx%2FPP-OCRv4%2Fcls%2Fch_ppocr_mobile_v2.0_cls_${tier}.onnx" &
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
            download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv4%2Frec%2Fjapan_PP-OCRv4_rec_mobile.onnx" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fjapan_PP-OCRv4_rec_mobile%2Fjapan_dict.txt" &
            pids+=($!)
            ;;
        chinese_cht|chinese-cht)
            download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv4%2Frec%2Fchinese_cht_PP-OCRv3_rec_mobile.onnx" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fchinese_cht_PP-OCRv3_rec_mobile%2Fchinese_cht_dict.txt" &
            pids+=($!)
            ;;
        kannada|ka)
            download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv4%2Frec%2Fka_PP-OCRv4_rec_mobile.onnx" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2Fkannada_PP-OCRv4_rec_mobile%2Fka_dict.txt" &
            pids+=($!)
            ;;
        *)
            download_file_to "$dir" "rec.onnx" "onnx%2FPP-OCRv4%2Frec%2F${lang}_PP-OCRv4_rec_mobile.onnx" &
            pids+=($!)
            download_file_to "$dir" "dict.txt" "paddle%2FPP-OCRv4%2Frec%2F${lang}_PP-OCRv4_rec_mobile%2F${lang}_dict.txt" &
            pids+=($!)
            ;;
    esac

    wait_jobs "${pids[@]}"
}

download_cls() {
    local dir="$1"
    echo "=== Downloading Text-Line Classifier (cls.onnx) ==="
    download_file_to "$dir" "cls.onnx" "onnx%2FPP-OCRv4%2Fcls%2Fch_ppocr_mobile_v2.0_cls_mobile.onnx"
}

download_orient() {
    local dir="$1"
    echo "=== Downloading Document Orientation Classifier (orient.onnx) ==="
    download_file_to "$dir" "orient.onnx" "https://huggingface.co/snowfluke/ppu-paddle-ocr-models/resolve/main/correction/PP-LCNet_x1_0_doc_ori.onnx"
}

download_unwarp() {
    local dir="$1"
    echo "=== Downloading Document Unwarp Model (uvdoc.onnx) ==="
    download_file_to "$dir" "uvdoc.onnx" "https://huggingface.co/PaddlePaddle/UVDoc_onnx/resolve/main/inference.onnx"
}

download_layout_s() {
    local dir="$1"
    echo "=== Downloading Document Layout Model Small (layout_s.onnx) ==="
    download_file_to "$dir" "layout_s.onnx" "https://www.modelscope.cn/api/v1/models/RapidAI/RapidDoc/repo?Revision=master&FilePath=layout%2FPP-DocLayout-S%2Fpp_doclayout_s.onnx"
}

download_layout_l() {
    local dir="$1"
    echo "=== Downloading Document Layout Model Large (layout_l.onnx) ==="
    download_file_to "$dir" "layout_l.onnx" "https://www.modelscope.cn/api/v1/models/RapidAI/RapidDoc/repo?Revision=master&FilePath=layout%2FPP-DocLayout_plus-L%2Fpp_doclayout_plus_l.onnx"
}

download_layout() {
    local dir="$1"
    local variant="${2:-s}"
    if [ "$variant" = "l" ] || [ "$variant" = "large" ] || [ "$variant" = "plus-l" ]; then
        download_layout_l "$dir"
    else
        download_layout_s "$dir"
    fi
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

    # Optional document pipeline models (both layout variants downloaded separately)
    start_model_group download_cls      "${OUTPUT_DIR:-$REPO_ROOT/models}"
    start_model_group download_orient   "${OUTPUT_DIR:-$REPO_ROOT/models}"
    start_model_group download_unwarp   "${OUTPUT_DIR:-$REPO_ROOT/models}"
    start_model_group download_layout_s "${OUTPUT_DIR:-$REPO_ROOT/models}"
    start_model_group download_layout_l "${OUTPUT_DIR:-$REPO_ROOT/models}"

    if ! wait_jobs "${tier_pids[@]}"; then
        tier_failed=1
    fi
    if [ "$tier_failed" -ne 0 ]; then
        echo "❌ One or more model groups failed" >&2
        exit 1
    fi
else
    TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v6}"
    case "$MODEL_TYPE" in
        ppocrv6)
            download_ppocrv6 "$TIER" "$TARGET_DIR"
            ;;
        ppocrv5)
            if [ -n "$LANG" ]; then
                TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5_$LANG}"
                download_ppocrv5_lang "$LANG" "$TARGET_DIR"
            else
                TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v5}"
                download_ppocrv5 "$TIER" "$TARGET_DIR"
            fi
            ;;
        ppocrv4)
            if [ -n "$LANG" ]; then
                TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4_$LANG}"
                download_ppocrv4_lang "$LANG" "$TARGET_DIR"
            else
                TARGET_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/PPOCR_v4}"
                download_ppocrv4 "$TIER" "$TARGET_DIR"
            fi
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac

    # Download requested optional models into the target model directory
    if [ "$WITH_CLS" = true ]; then
        download_cls "$TARGET_DIR"
    fi
    if [ "$WITH_ORIENT" = true ]; then
        download_orient "$TARGET_DIR"
    fi
    if [ "$WITH_UNWARP" = true ]; then
        download_unwarp "$TARGET_DIR"
    fi
    if [ "$WITH_LAYOUT" = true ]; then
        download_layout_s "$TARGET_DIR"
        download_layout_l "$TARGET_DIR"
    fi
fi

echo ""
echo "✅ Download complete"

if [ "$CONVERT_RTEN" = true ]; then
    echo "=== Converting downloaded models to true .rten format and removing .onnx ==="
    RTEN_CONVERT=""
    if command -v rten-convert &>/dev/null; then
        RTEN_CONVERT="rten-convert"
    elif [ -f "/tmp/rten-env/bin/rten-convert" ]; then
        RTEN_CONVERT="/tmp/rten-env/bin/rten-convert"
    elif [ -f "$HOME/.local/bin/rten-convert" ]; then
        RTEN_CONVERT="$HOME/.local/bin/rten-convert"
    else
        python3 -m pip install --upgrade rten-convert --break-system-packages 2>/dev/null || python3 -m pip install rten-convert 2>/dev/null || true
        RTEN_CONVERT="$(command -v rten-convert || true)"
    fi

    if [ -n "$RTEN_CONVERT" ]; then
        target_root="${OUTPUT_DIR:-$REPO_ROOT/models}"
        find "$target_root" -name "*.onnx" | while read -r onnx_path; do
            rten_path="${onnx_path%.onnx}.rten"
            echo "Converting $onnx_path -> $rten_path..."
            "$RTEN_CONVERT" "$onnx_path" "$rten_path"
            if [ "$KEEP_ONNX" = false ]; then
                rm -f "$onnx_path"
            fi
        done
        echo "✓ All ONNX models converted to .rten"
    else
        echo "⚠️ rten-convert could not be found or installed, skipping conversion" >&2
    fi
fi

