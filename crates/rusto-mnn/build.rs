use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to find repo root");

    let mnn_root = repo_root.join("vendor/alibaba-mnn");
    let mnn_include = mnn_root.join("include");
    let mnn_build = mnn_root.join("build");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&mnn_include)
        .include("csrc")
        .file("csrc/mnn_wrapper.cpp")
        .compile("rusto_mnn_wrapper");

    println!("cargo:rustc-link-search=native={}", mnn_build.display());
    println!("cargo:rustc-link-lib=static=MNN");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }

    println!("cargo:rerun-if-changed=csrc/mnn_wrapper.h");
    println!("cargo:rerun-if-changed=csrc/mnn_wrapper.cpp");
}

