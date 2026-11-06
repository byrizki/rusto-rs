use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn has_static_mnn(dir: &Path) -> bool {
    dir.join("libMNN.a").exists() || dir.join("MNN.lib").exists()
}

fn build_mnn_with_cmake(mnn_root: &Path, out_dir: &Path) -> PathBuf {
    let build_dir = out_dir.join("mnn-build");
    if has_static_mnn(&build_dir) {
        return build_dir;
    }
    std::fs::create_dir_all(&build_dir).expect("Failed to create MNN CMake build directory");

    let status = Command::new("cmake")
        .arg(mnn_root)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DMNN_BUILD_SHARED_LIBS=OFF")
        .arg("-DMNN_SEP_BUILD=OFF")
        .arg("-DMNN_BUILD_TOOLS=OFF")
        .arg("-DMNN_BUILD_DEMO=OFF")
        .arg("-DMNN_BUILD_TRAIN=OFF")
        .arg("-DMNN_BUILD_CONVERTER=OFF")
        .arg("-DMNN_PORTABLE_BUILD=ON")
        .current_dir(&build_dir)
        .status()
        .expect("Failed to invoke cmake to configure Alibaba MNN");
    assert!(status.success(), "CMake configure failed for Alibaba MNN");

    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());
    let status = Command::new("cmake")
        .args(["--build", ".", "--target", "MNN", "-j", &jobs])
        .current_dir(&build_dir)
        .status()
        .expect("Failed to invoke cmake --build for Alibaba MNN");
    assert!(status.success(), "CMake build failed for Alibaba MNN");

    build_dir
}

fn main() {
    println!("cargo:rerun-if-env-changed=MNN_DIR");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mnn_root = env::var("MNN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            manifest_dir
                .parent()
                .and_then(|p| p.parent())
                .expect("Failed to find repo root")
                .join("vendor/alibaba-mnn")
        });

    let mnn_include = mnn_root.join("include");
    if !mnn_include.exists() {
        panic!(
            "Alibaba MNN headers not found at {}. Initialize git submodules (`git submodule update --init --recursive`) or set MNN_DIR.",
            mnn_include.display()
        );
    }

    let prebuilt_dir = mnn_root.join("build");
    let mnn_lib_dir = if has_static_mnn(&prebuilt_dir) {
        prebuilt_dir
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        build_mnn_with_cmake(&mnn_root, &out_dir)
    };

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&mnn_include)
        .include("csrc")
        .file("csrc/mnn_wrapper.cpp")
        .compile("rusto_mnn_wrapper");

    println!("cargo:rustc-link-search=native={}", mnn_lib_dir.display());
    println!("cargo:rustc-link-lib=static=MNN");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os != "windows" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }

    println!("cargo:rerun-if-changed=csrc/mnn_wrapper.h");
    println!("cargo:rerun-if-changed=csrc/mnn_wrapper.cpp");
}

