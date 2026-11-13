use std::path::PathBuf;

use rusto_http::{build_state, router, Config};

/// `OCR_MODELS_DIR` holds one subdirectory per model (tiny/, small/, ...), each with
/// det.onnx, rec.onnx and dict.txt. `OCR_MODEL_DIR` serves a single model named "tiny".
fn discover_models(env: &impl Fn(&str) -> Option<String>) -> Vec<(String, PathBuf)> {
    if let Some(root) = env("OCR_MODELS_DIR") {
        let mut found: Vec<_> = std::fs::read_dir(&root)
            .unwrap_or_else(|e| {
                eprintln!("error: cannot read OCR_MODELS_DIR {root}: {e}");
                std::process::exit(1);
            })
            .filter_map(Result::ok)
            .filter(|e| e.path().join("det.onnx").is_file())
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .collect();
        found.sort();
        return found;
    }
    vec![(
        "tiny".into(),
        env("OCR_MODEL_DIR")
            .unwrap_or_else(|| "models/tiny".into())
            .into(),
    )]
}

#[tokio::main]
async fn main() {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());

    let models = discover_models(&env);
    let default_model = env("OCR_DEFAULT_MODEL")
        .or_else(|| {
            models
                .iter()
                .find(|(n, _)| n == "tiny")
                .map(|(n, _)| n.clone())
        })
        .or_else(|| models.first().map(|(n, _)| n.clone()))
        .unwrap_or_default();
    let cfg = Config {
        models,
        default_model,
        pool_size: env("OCR_POOL_SIZE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2),
        api_key: env("OCR_API_KEY"),
        allowed_hosts: env("OCR_ALLOWED_HOSTS")
            .map(|v| {
                v.split(',')
                    .map(|h| h.trim().to_owned())
                    .filter(|h| !h.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        autorotate: env("OCR_AUTOROTATE")
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on")),
    };
    let addr = env("OCR_LISTEN").unwrap_or_else(|| "127.0.0.1:8895".into());
    let names: Vec<_> = cfg.models.iter().map(|(n, _)| n.clone()).collect();

    let state = build_state(cfg).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("error: cannot listen on {addr}: {e}");
            std::process::exit(1);
        });
    eprintln!(
        "rusto-server listening on http://{addr}  (POST /v1/ocr)  models: {}",
        names.join(", ")
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .unwrap();
}
