#!/bin/bash

# Script to copy model files for bundling with React Native packages
# Run from repository root: ./packages/react-native/scripts/copy-models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_VERSION="${1:-PPOCR_v6}"
if [ -d "$REPO_ROOT/models/$MODEL_VERSION" ]; then
    MODELS_SOURCE="$REPO_ROOT/models/$MODEL_VERSION"
elif [ -d "$REPO_ROOT/models/PPOCR_v6" ]; then
    MODELS_SOURCE="$REPO_ROOT/models/PPOCR_v6"
else
    MODELS_SOURCE="$REPO_ROOT/models/PPOCR_v5"
fi
RN_PACKAGE="$REPO_ROOT/packages/react-native"
ANDROID_PACKAGE="$REPO_ROOT/packages/android"

echo "Copying model files for React Native bundling..."
echo "Repository root: $REPO_ROOT"
echo "Model source: $MODELS_SOURCE"
echo ""

# Check if source models exist; if not, download on the fly
if [ ! -f "$MODELS_SOURCE/det.mnn" ] || [ ! -f "$MODELS_SOURCE/rec.mnn" ] || [ ! -f "$MODELS_SOURCE/dict.txt" ]; then
    echo "Models not found in $MODELS_SOURCE. Downloading default PP-OCRv6 tiny models..."
    bash "$REPO_ROOT/scripts/download_models.sh" --output-dir "$MODELS_SOURCE"
fi

# Android: Copy to main android package assets and react-native android assets
echo "📦 Android Setup..."
ANDROID_ASSETS="$ANDROID_PACKAGE/src/main/assets"
RN_ANDROID_ASSETS="$RN_PACKAGE/android/src/main/assets"
mkdir -p "$ANDROID_ASSETS"
mkdir -p "$RN_ANDROID_ASSETS"

cp "$MODELS_SOURCE/det.mnn" "$ANDROID_ASSETS/"
cp "$MODELS_SOURCE/rec.mnn" "$ANDROID_ASSETS/"
cp "$MODELS_SOURCE/dict.txt" "$ANDROID_ASSETS/"

cp "$MODELS_SOURCE/det.mnn" "$RN_ANDROID_ASSETS/"
cp "$MODELS_SOURCE/rec.mnn" "$RN_ANDROID_ASSETS/"
cp "$MODELS_SOURCE/dict.txt" "$RN_ANDROID_ASSETS/"

echo "✓ Copied models to $ANDROID_ASSETS and $RN_ANDROID_ASSETS"
echo "  - det.mnn ($(du -h "$ANDROID_ASSETS/det.mnn" | cut -f1))"
echo "  - rec.mnn ($(du -h "$ANDROID_ASSETS/rec.mnn" | cut -f1))"
echo "  - dict.txt ($(du -h "$ANDROID_ASSETS/dict.txt" | cut -f1))"
echo ""

# iOS: Copy to react-native ios/models directory
echo "🍎 iOS Setup..."
IOS_MODELS="$RN_PACKAGE/ios/models"
mkdir -p "$IOS_MODELS"

cp "$MODELS_SOURCE/det.mnn" "$IOS_MODELS/"
cp "$MODELS_SOURCE/rec.mnn" "$IOS_MODELS/"
cp "$MODELS_SOURCE/dict.txt" "$IOS_MODELS/"

echo "✓ Copied models to $IOS_MODELS"
echo "  - det.mnn ($(du -h "$IOS_MODELS/det.mnn" | cut -f1))"
echo "  - rec.mnn ($(du -h "$IOS_MODELS/rec.mnn" | cut -f1))"
echo "  - dict.txt ($(du -h "$IOS_MODELS/dict.txt" | cut -f1))"
echo ""

# Calculate total size
TOTAL_ANDROID=$(du -sh "$ANDROID_ASSETS" | cut -f1)
TOTAL_IOS=$(du -sh "$IOS_MODELS" | cut -f1)

echo "✅ Model files copied successfully!"
echo ""
echo "Total bundled size:"
echo "  Android: $TOTAL_ANDROID"
echo "  iOS: $TOTAL_IOS"
echo ""
echo "Next steps:"
echo "  1. Build Android: cd packages/react-native/android && ./gradlew assembleRelease"
echo "  2. Build iOS: cd packages/react-native/ios && pod install"
echo "  3. Use in app: await initialize() // No parameters needed!"
