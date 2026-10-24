use std::path::PathBuf;

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    let config_path = crate_dir.join("cbindgen.toml");
    let config = cbindgen::Config::from_file(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read cbindgen config at {:?}: {e}", config_path));

    let output_path = crate_dir.join("include").join("rusto.h");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(&output_path);

    println!("Successfully generated C header at: {}", output_path.display());
}
