#!/bin/bash
# Script to download default bundled OCR models on the fly
# Priority:
# 1. https://github.com/byrizki/rusto-rs-models/releases
# 2. https://www.modelscope.cn/models/RapidAI/RapidOCR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/models/PPOCR_v6"
TIER="tiny"
VERSION="v1.0.0"

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --tier) TIER="$2"; shift ;;
        --version) VERSION="$2"; shift ;;
        *) OUTPUT_DIR="$1" ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR"

echo "=== Downloading PP-OCRv6 ($TIER tier) models ==="
echo "Destination: $OUTPUT_DIR"

GH_RELEASE_URL="https://github.com/byrizki/rusto-rs-models/releases/download/$VERSION"
GH_LATEST_URL="https://github.com/byrizki/rusto-rs-models/releases/latest/download"
MODELSCOPE_BASE="https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath="

download_file() {
    local filename="$1"
    local gh_name="$2"
    local ms_path="$3"
    local dest="$OUTPUT_DIR/$filename"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ Already exists: $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    echo "Downloading $filename..."
    
    # 1. Try specific GitHub release
    if curl -sLf "$GH_RELEASE_URL/$gh_name" -o "$dest" 2>/dev/null && [ -s "$dest" ]; then
        echo "✓ Downloaded from GitHub release ($VERSION): $filename"
        return 0
    fi

    # 2. Try latest GitHub release
    if curl -sLf "$GH_LATEST_URL/$gh_name" -o "$dest" 2>/dev/null && [ -s "$dest" ]; then
        echo "✓ Downloaded from GitHub release (latest): $filename"
        return 0
    fi

    # 3. Fallback to ModelScope
    echo "  Downloading from ModelScope fallback..."
    if curl -L -f "$MODELSCOPE_BASE$ms_path" -o "$dest"; then
        echo "✓ Downloaded from ModelScope: $filename"
        return 0
    fi

    echo "❌ Failed to download $filename"
    return 1
}

# Download detection model
download_file "det.mnn" "ppocrv6_det_${TIER}.mnn" "mnn%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_${TIER}.mnn"

# Download recognition model
download_file "rec.mnn" "ppocrv6_rec_${TIER}.mnn" "mnn%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_${TIER}.mnn"

# Download dictionary
if [ "$TIER" = "tiny" ]; then
    download_file "dict.txt" "ppocrv6_tiny_dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny%2Fppocrv6_tiny_dict.txt"
else
    download_file "dict.txt" "ppocrv6_dict.txt" "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_small%2Fppocrv6_dict.txt"
fi

echo ""
echo "✅ Models ready in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
