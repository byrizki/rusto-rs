use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=mnn_wrapper.h");
    println!("cargo:rerun-if-changed=mnn_wrapper.cpp");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Resolve MNN headers
    let include_path = find_include_dir();

    // Resolve or build MNN static library
    let lib_dir = find_or_build_mnn(&target_os, &target_arch, &out_dir);

    // Compile our C wrapper
    cc::Build::new()
        .cpp(true)
        .file("mnn_wrapper.cpp")
        .include(&include_path)
        .flag_if_supported("-std=c++11")
        .compile("mnn_wrapper");

    // Link MNN library
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=MNN");

    // Link C++ standard library
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "windows" {
        // MSVC links CRT automatically
    } else if target_os == "android" {
        println!("cargo:rustc-link-lib=c++_shared");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Feature-based linking
    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    if cfg!(feature = "coreml") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    if cfg!(feature = "opencl") {
        if target_os == "macos" || target_os == "ios" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }

    if cfg!(feature = "vulkan") {
        println!("cargo:rustc-link-lib=vulkan");
    }

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }

    if target_os == "android" {
        println!("cargo:rustc-link-lib=log");
        if cfg!(feature = "nnapi") {
            println!("cargo:rustc-link-lib=android");
        }
    }

    // Generate bindings for our C wrapper
    let mut builder = bindgen::Builder::default()
        .header("mnn_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    // Fix for iOS Simulator build error where bindgen passes invalid target triple
    if env::var("TARGET").unwrap_or_default() == "aarch64-apple-ios-sim" {
        builder = builder.clang_arg("--target=arm64-apple-ios-simulator");
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn find_include_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Check local crate include/ (bundled in crate package)
    let local_include = manifest_dir.join("include");
    if local_include.join("MNN/Interpreter.hpp").exists() {
        return local_include;
    }

    // 2. Check workspace vendor/mnn/include
    let mut current = manifest_dir.as_path();
    while let Some(parent) = current.parent() {
        let vendor_include = parent.join("vendor/mnn/include");
        if vendor_include.join("MNN/Interpreter.hpp").exists() {
            return vendor_include;
        }
        current = parent;
    }

    // Fallback to local include
    local_include
}

fn find_or_build_mnn(target_os: &str, target_arch: &str, out_dir: &Path) -> PathBuf {
    let lib_name = if target_os == "windows" { "MNN.lib" } else { "libMNN.a" };

    // 1. Explicit override via MNN_LIB_DIR or MNN_DIR
    if let Ok(dir_str) = env::var("MNN_LIB_DIR").or_else(|_| env::var("MNN_DIR")) {
        let dir = PathBuf::from(&dir_str);
        if dir.join(lib_name).exists() {
            return dir;
        }
        let lib_subdir = dir.join("lib");
        if lib_subdir.join(lib_name).exists() {
            return lib_subdir;
        }
    }

    // 2. If in local workspace with vendor/mnn and not explicitly targeting prebuilt only, build/use local
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut is_local_workspace = false;
    let mut current = manifest_dir.as_path();
    while let Some(parent) = current.parent() {
        if parent.join("vendor/mnn/CMakeLists.txt").exists() {
            is_local_workspace = true;
            break;
        }
        current = parent;
    }

    let force_source = cfg!(feature = "build-from-source");

    // 3. Try prebuilt download if not forced to build from source and not developing in local workspace with vendor/mnn
    if !force_source && !is_local_workspace {
        if let Some(lib_dir) = try_get_prebuilt(out_dir, lib_name) {
            return lib_dir;
        }
    }

    // 4. Build from source
    build_from_source(target_os, target_arch, out_dir, lib_name)
}

fn try_get_prebuilt(out_dir: &Path, lib_name: &str) -> Option<PathBuf> {
    let target = env::var("TARGET").unwrap_or_default();
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.3".to_string());
    let prebuilt_dir = out_dir.join("prebuilt");
    let lib_dest = prebuilt_dir.join(lib_name);

    if lib_dest.exists() {
        return Some(prebuilt_dir);
    }

    // List of prebuilt supported targets
    let supported = matches!(
        target.as_str(),
        "x86_64-unknown-linux-gnu"
            | "aarch64-unknown-linux-gnu"
            | "x86_64-apple-darwin"
            | "aarch64-apple-darwin"
            | "x86_64-pc-windows-msvc"
    );

    if !supported {
        println!("cargo:warning=Target {} has no official prebuilt MNN binary, will build from source.", target);
        return None;
    }

    let archive_name = format!("mnn-static-{}.tar.gz", target);
    let url = format!(
        "https://github.com/byrizki/rusto-rs/releases/download/v{}/{}",
        version, archive_name
    );

    let _ = fs::create_dir_all(&prebuilt_dir);
    let archive_path = out_dir.join(&archive_name);

    println!("cargo:warning=Downloading prebuilt MNN for {} from {}...", target, url);

    if download_file(&url, &archive_path) && archive_path.exists() {
        let extract_status = Command::new("tar")
            .args(&["-xzf", archive_path.to_str().unwrap(), "-C", prebuilt_dir.to_str().unwrap()])
            .status();

        if let Ok(status) = extract_status {
            if status.success() && (lib_dest.exists() || search_for_lib(&prebuilt_dir, lib_name).is_some()) {
                let final_dir = search_for_lib(&prebuilt_dir, lib_name).unwrap_or(prebuilt_dir);
                println!("cargo:warning=Successfully extracted prebuilt MNN to {}", final_dir.display());
                return Some(final_dir);
            }
        }
    }

    println!("cargo:warning=Prebuilt binary not accessible from {}, falling back to local source build.", url);
    None
}

fn download_file(url: &str, dest: &Path) -> bool {
    // Try curl (standard on Linux, macOS, and modern Windows 10/11)
    let status = Command::new("curl")
        .args(&["-sSL", "-f", "-o", dest.to_str().unwrap(), url])
        .status();

    if let Ok(s) = status {
        if s.success() {
            return true;
        }
    }

    // Windows fallback: PowerShell Invoke-WebRequest
    if cfg!(target_os = "windows") {
        let ps_cmd = format!(
            "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '{}' -OutFile '{}'",
            url, dest.display()
        );
        let status = Command::new("powershell")
            .args(&["-NoProfile", "-NonInteractive", "-Command", &ps_cmd])
            .status();
        if let Ok(s) = status {
            return s.success();
        }
    }

    false
}

fn find_mnn_source_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Search parent directories for vendor/mnn
    let mut current = manifest_dir.as_path();
    while let Some(parent) = current.parent() {
        let candidate = parent.join("vendor/mnn");
        if candidate.join("CMakeLists.txt").exists() {
            return candidate;
        }
        current = parent;
    }

    // If not found, clone MNN repo into OUT_DIR
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let clone_dir = out_dir.join("mnn-src");
    if !clone_dir.join("CMakeLists.txt").exists() {
        println!("cargo:warning=MNN source not found locally, cloning from GitHub...");
        let status = Command::new("git")
            .args(&[
                "clone",
                "--depth", "1",
                "--branch", "2.8.3",
                "https://github.com/alibaba/MNN.git",
                clone_dir.to_str().unwrap(),
            ])
            .status();

        if let Ok(s) = status {
            if s.success() {
                return clone_dir;
            }
        }
        panic!("MNN source directory not found and failed to clone MNN repository from GitHub.");
    }
    clone_dir
}

fn build_from_source(target_os: &str, target_arch: &str, _out_dir: &Path, lib_name: &str) -> PathBuf {
    let mnn_dir = find_mnn_source_dir();

    let has_metal = cfg!(feature = "metal");
    let has_coreml = cfg!(feature = "coreml");
    let has_opencl = cfg!(feature = "opencl");
    let has_vulkan = cfg!(feature = "vulkan");
    let has_cuda = cfg!(feature = "cuda");
    let has_nnapi = cfg!(feature = "nnapi");
    let has_hiai = cfg!(feature = "hiai");
    let has_qnn = cfg!(feature = "qnn");

    let mut cmake_config = cmake::Config::new(&mnn_dir);

    cmake_config
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_BUILD_TRAIN", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_EVALUATION", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_ARM82", if target_arch == "aarch64" { "ON" } else { "OFF" })
        .define("MNN_METAL", if has_metal { "ON" } else { "OFF" })
        .define("MNN_COREML", if has_coreml { "ON" } else { "OFF" })
        .define("MNN_OPENCL", if has_opencl { "ON" } else { "OFF" })
        .define("MNN_VULKAN", if has_vulkan { "ON" } else { "OFF" })
        .define("MNN_CUDA", if has_cuda { "ON" } else { "OFF" })
        .define("MNN_HIAI", if has_hiai { "ON" } else { "OFF" })
        .build_target("MNN");

    if has_nnapi {
        cmake_config.define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON");
    }

    if has_qnn {
        cmake_config.define("MNN_SUPPORT_QNN", "ON");
    }

    if let Ok(num) = env::var("NUM_JOBS") {
        cmake_config.define("CMAKE_BUILD_PARALLEL_LEVEL", num);
    }

    if target_os == "android" {
        cmake_config
            .define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON")
            .define("MNN_AVX2", "OFF")
            .define("MNN_AVX512", "OFF")
            .define("MNN_FMA", "OFF");

        if let Ok(ndk_path) = env::var("ANDROID_NDK_HOME").or_else(|_| env::var("NDK_HOME")) {
            let toolchain = PathBuf::from(&ndk_path).join("build/cmake/android.toolchain.cmake");
            if toolchain.exists() {
                cmake_config.define("CMAKE_TOOLCHAIN_FILE", toolchain);
                let abi = match target_arch {
                    "aarch64" => "arm64-v8a",
                    "arm" => "armeabi-v7a",
                    "x86" => "x86",
                    "x86_64" => "x86_64",
                    _ => "arm64-v8a",
                };
                cmake_config.define("ANDROID_ABI", abi);
                cmake_config.define("ANDROID_PLATFORM", "android-21");
                cmake_config.define("ANDROID_STL", "c++_shared");
            }
        }
    }

    if target_os == "macos" {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_string());
        cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
        cmake_config.cxxflag("-include cstdint");
    }
    if target_os == "ios" {
        let deployment_target = env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "12.0".to_string());
        cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
        cmake_config.cxxflag("-include cstdint");
    }

    let dst = cmake_config.build();

    if let Some(found_dir) = search_for_lib(&dst, lib_name) {
        return found_dir;
    }

    // Default fallback path in CMake output
    dst.join("build")
}

fn search_for_lib(root: &Path, lib_name: &str) -> Option<PathBuf> {
    if let Ok(entries) = walkdir::WalkDir::new(root).into_iter().collect::<Result<Vec<_>, _>>() {
        for entry in entries {
            if entry.file_name() == lib_name {
                if let Some(parent) = entry.path().parent() {
                    return Some(parent.to_path_buf());
                }
            }
        }
    }
    None
}

