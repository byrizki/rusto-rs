use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=mnn_wrapper.h");
    println!("cargo:rerun-if-changed=mnn_wrapper.cpp");

    // Get the MNN vendor directory
    let mnn_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vendor/mnn");

    // Check enabled features
    let has_metal = cfg!(feature = "metal");
    let has_coreml = cfg!(feature = "coreml");
    let has_opencl = cfg!(feature = "opencl");
    let has_vulkan = cfg!(feature = "vulkan");
    let has_cuda = cfg!(feature = "cuda");
    let has_nnapi = cfg!(feature = "nnapi");
    let has_hiai = cfg!(feature = "hiai");
    let has_qnn = cfg!(feature = "qnn");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Build MNN using CMake with optimizations for faster builds
    let mut cmake_config = cmake::Config::new(&mnn_dir);
    
    cmake_config
        // Disable unnecessary components for faster builds
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
        // Enable ARM optimizations if applicable
        .define("MNN_ARM82", if target_arch == "aarch64" { "ON" } else { "OFF" })
        // Feature-based backends
        .define("MNN_METAL", if has_metal { "ON" } else { "OFF" })
        .define("MNN_COREML", if has_coreml { "ON" } else { "OFF" })
        .define("MNN_OPENCL", if has_opencl { "ON" } else { "OFF" })
        .define("MNN_VULKAN", if has_vulkan { "ON" } else { "OFF" })
        .define("MNN_CUDA", if has_cuda { "ON" } else { "OFF" })
        .define("MNN_HIAI", if has_hiai { "ON" } else { "OFF" })
        .build_target("MNN");
    
    // Platform-specific backends
    if has_nnapi {
        cmake_config.define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON");
    }
    
    if has_qnn {
        cmake_config.define("MNN_SUPPORT_QNN", "ON");
    }
    
    // Enable parallel build
    if let Ok(num) = env::var("NUM_JOBS") {
        cmake_config.define("CMAKE_BUILD_PARALLEL_LEVEL", num);
    }

    // Configure for Android cross-compilation
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
                let abi = match target_arch.as_str() {
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

    // Pin macOS/iOS deployment targets so that building on a recent macOS host
    // (e.g. macOS 26) doesn't propagate an unsupported -mmacosx-version-min
    // flag into MNN's CMake build.
    if target_os == "macos" {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "11.0".to_string());
        cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
    }
    if target_os == "ios" {
        let deployment_target = env::var("IPHONEOS_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "12.0".to_string());
        cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
    }
    
    let dst = cmake_config.build();

    // Compile our C wrapper
    let include_path = mnn_dir.join("include");
    cc::Build::new()
        .cpp(true)
        .file("mnn_wrapper.cpp")
        .include(&include_path)
        .flag_if_supported("-std=c++11")
        .compile("mnn_wrapper");

    // Tell cargo to look for the library
    // Recursively search for the MNN library file to handle different build layouts (e.g. MSVC Release/Debug)
    let lib_name = if target_os == "windows" { "MNN.lib" } else { "libMNN.a" };
    let mut found_lib = false;
    
    if let Ok(entries) = walkdir::WalkDir::new(&dst).into_iter().collect::<Result<Vec<_>, _>>() {
        for entry in entries {
            if entry.file_name() == lib_name {
                if let Some(parent) = entry.path().parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                    found_lib = true;
                    // Don't break, allow adding multiple candidates if duplicates exist (rare but safe)
                }
            }
        }
    }
    
    if !found_lib {
        // Fallback paths if walkdir fails or library not found (e.g. not built yet?)
        println!("cargo:warning=MNN library ({}) not found in build output search. Using default fallback paths.", lib_name);
        println!("cargo:rustc-link-search=native={}/build", dst.display());
        println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
        println!("cargo:rustc-link-search=native={}/build/Debug", dst.display());
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    }

    // Link MNN library
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
    if has_metal {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }
    
    if has_coreml {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    
    if has_opencl {
        if target_os == "macos" || target_os == "ios" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }
    
    if has_vulkan {
        println!("cargo:rustc-link-lib=vulkan");
    }
    
    if has_cuda {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
    
    if target_os == "android" {
        println!("cargo:rustc-link-lib=log");
        if has_nnapi {
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

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
