#!/bin/bash
set -e

export ANDROID_NDK_HOME=/mnt/development/Android/Sdk/ndk/26.1.10909125
export ANDROID_NDK=$ANDROID_NDK_HOME
export NDK_HOME=$ANDROID_NDK_HOME
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake

export CC_aarch64_linux_android=aarch64-linux-android21-clang
export CXX_aarch64_linux_android=aarch64-linux-android21-clang++
export AR_aarch64_linux_android=llvm-ar

export CC_armv7_linux_androideabi=armv7a-linux-androideabi21-clang
export CXX_armv7_linux_androideabi=armv7a-linux-androideabi21-clang++
export AR_armv7_linux_androideabi=llvm-ar

export CC_i686_linux_android=i686-linux-android21-clang
export CXX_i686_linux_android=i686-linux-android21-clang++
export AR_i686_linux_android=llvm-ar

export CC_x86_64_linux_android=x86_64-linux-android21-clang
export CXX_x86_64_linux_android=x86_64-linux-android21-clang++
export AR_x86_64_linux_android=llvm-ar

echo "=== Building aarch64-linux-android ==="
cargo build --release --target aarch64-linux-android --features ffi

echo "=== Building armv7-linux-androideabi ==="
cargo build --release --target armv7-linux-androideabi --features ffi

echo "=== Building x86_64-linux-android ==="
cargo build --release --target x86_64-linux-android --features ffi

echo "=== Creating jniLibs directories ==="
mkdir -p packages/android/src/main/jniLibs/{arm64-v8a,armeabi-v7a,x86_64}
cp target/aarch64-linux-android/release/librusto.so packages/android/src/main/jniLibs/arm64-v8a/
cp target/armv7-linux-androideabi/release/librusto.so packages/android/src/main/jniLibs/armeabi-v7a/
cp target/x86_64-linux-android/release/librusto.so packages/android/src/main/jniLibs/x86_64/

echo "=== Copying assets ==="
mkdir -p packages/android/src/main/assets
mkdir -p packages/react-native/android/src/main/assets
cp models/PPOCR_v5/det.mnn packages/android/src/main/assets/
cp models/PPOCR_v5/rec.mnn packages/android/src/main/assets/
cp models/PPOCR_v5/dict.txt packages/android/src/main/assets/
cp models/PPOCR_v5/det.mnn packages/react-native/android/src/main/assets/
cp models/PPOCR_v5/rec.mnn packages/react-native/android/src/main/assets/
cp models/PPOCR_v5/dict.txt packages/react-native/android/src/main/assets/

echo "=== Building Android AAR ==="
cd packages/android
chmod +x gradlew
./gradlew assembleRelease

echo "=== Copying AAR to React Native package ==="
cd ../..
mkdir -p packages/react-native/android/libs
cp packages/android/build/outputs/aar/*.aar packages/react-native/android/libs/RustO.aar

echo "=== Done! ==="
