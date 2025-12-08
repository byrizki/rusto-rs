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
        .define("MNN_ARM82", if cfg!(target_arch = "aarch64") { "ON" } else { "OFF" })
        // Feature-based backends
        .define("MNN_METAL", if has_metal { "ON" } else { "OFF" })
        .define("MNN_COREML", if has_coreml { "ON" } else { "OFF" })
        .define("MNN_OPENCL", if has_opencl { "ON" } else { "OFF" })
        .define("MNN_VULKAN", if has_vulkan { "ON" } else { "OFF" })
        .define("MNN_CUDA", if has_cuda { "ON" } else { "OFF" })
        .define("MNN_HIAI", if has_hiai { "ON" } else { "OFF" })
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
    // MNN build output is in build directory since we skip install
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/lib", dst.display()); // Keep these as fallbacks
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    
    // Link MNN library
    println!("cargo:rustc-link-lib=static=MNN");
    
    // Link C++ standard library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
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
        if cfg!(target_os = "macos") {
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
    
    if cfg!(target_os = "android") {
        println!("cargo:rustc-link-lib=log");
        if has_nnapi {
            println!("cargo:rustc-link-lib=android");
        }
    }

    // Generate bindings for our C wrapper
    let bindings = bindgen::Builder::default()
        .header("mnn_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
