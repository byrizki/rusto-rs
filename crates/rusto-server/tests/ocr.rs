use std::path::PathBuf;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use http_body_util::BodyExt;
use rusto_http::{build_state, router, Config};
use tower::ServiceExt;

/// OCR_MODELS_DIR holds one subdirectory per model (tiny/, small/, ...).
fn models() -> Option<Vec<(String, PathBuf)>> {
    let root: PathBuf = std::env::var("OCR_MODELS_DIR").ok()?.into();
    let mut found: Vec<_> = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.path().join("det.onnx").is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    found.sort();
    (!found.is_empty()).then_some(found)
}

fn app(api_key: Option<&str>) -> Option<axum::Router> {
    let state = build_state(Config {
        models: models()?,
        default_model: "tiny".into(),
        pool_size: 1,
        api_key: api_key.map(String::from),
        allowed_hosts: vec![],
        autorotate: false,
    })
    .expect("state");
    Some(router(state))
}

async fn post(
    app: axum::Router,
    auth: Option<&str>,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::post("/v1/ocr").header("content-type", "application/json");
    if let Some(a) = auth {
        req = req.header("authorization", a);
    }
    let resp = app
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

fn data_uri(path: &str) -> String {
    let bytes = std::fs::read(path).expect("fixture");
    format!("data:image/png;base64,{}", STANDARD.encode(bytes))
}

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/text.png");

#[tokio::test]
async fn ocr_returns_mistral_shape() {
    let Some(app) = app(None) else {
        eprintln!("OCR_MODELS_DIR not set; skipping");
        return;
    };
    let (status, v) = post(
        app,
        None,
        serde_json::json!({
            "model": "mistral-ocr-latest",
            "document": {"type": "image_url", "image_url": data_uri(FIXTURE)}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["model"], "rusto-ppocrv6-tiny");
    assert_eq!(v["usage_info"]["pages_processed"], 1);
    let page = &v["pages"][0];
    assert_eq!(page["index"], 0);
    assert!(page["dimensions"]["width"].as_u64().unwrap() > 0);
    let md = page["markdown"].as_str().unwrap();
    assert!(!md.trim().is_empty(), "expected recognized text, got {v}");
    eprintln!("markdown: {md}");
}

#[tokio::test]
async fn object_form_url_and_auth() {
    let Some(app) = app(Some("secret")) else {
        return;
    };
    let body = serde_json::json!({"document": {"type": "image_url", "image_url": {"url": data_uri(FIXTURE)}}});
    let (status, _) = post(app.clone(), None, body.clone()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, v) = post(app, Some("Bearer secret"), body).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn rejects_bad_input() {
    let Some(app) = app(None) else { return };
    let (s, v) = post(app.clone(), None, serde_json::json!({"document": {"type": "image_url", "image_url": "https://example.com/a.png"}})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["object"], "error");
    let pdf = format!(
        "data:application/pdf;base64,{}",
        STANDARD.encode(b"%PDF-1.4 x")
    );
    let (s, _) = post(
        app,
        None,
        serde_json::json!({"document": {"type": "document_url", "document_url": pdf}}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn selects_model_per_request() {
    let Some(app) = app(None) else { return };
    let have_small = models().unwrap().iter().any(|(n, _)| n == "small");
    let doc = serde_json::json!({"type": "image_url", "image_url": data_uri(FIXTURE)});

    let (s, v) = post(
        app.clone(),
        None,
        serde_json::json!({"model": "tiny", "document": doc}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["model"], "rusto-ppocrv6-tiny");

    if have_small {
        let (s, v) = post(
            app.clone(),
            None,
            serde_json::json!({"model": "small", "document": doc}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["model"], "rusto-ppocrv6-small");
        assert!(!v["pages"][0]["markdown"]
            .as_str()
            .unwrap()
            .trim()
            .is_empty());
    }

    let (s, v) = post(
        app,
        None,
        serde_json::json!({"model": "no-such-model", "document": doc}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["message"].as_str().unwrap().contains("available:"), "{v}");
}

#[tokio::test]
async fn exif_orientation_is_applied_before_ocr() {
    let Some(app) = app(None) else {
        eprintln!("OCR_MODELS_DIR not set; skipping");
        return;
    };
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/text-rot6.jpg"))
        .expect("fixture");
    let uri = format!("data:image/jpeg;base64,{}", STANDARD.encode(bytes));
    let (status, v) = post(app, None, serde_json::json!({"autorotate": true, "document": {"type": "image_url", "image_url": uri}})).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let page = &v["pages"][0];
    assert_eq!(page["orientation_corrected"], true);
    // The stored pixels are 160x700 (portrait); upright they are 700x160.
    assert_eq!(page["dimensions"]["width"], 700, "{page}");
    assert_eq!(page["dimensions"]["height"], 160, "{page}");
    assert!(
        !page["markdown"].as_str().unwrap().trim().is_empty(),
        "no text read after rotation: {page}"
    );
    assert!(page["text_angle"].as_f64().unwrap().abs() < 5.0, "{page}");
}

#[tokio::test]
async fn autorotate_is_opt_in() {
    let Some(app) = app(None) else {
        eprintln!("OCR_MODELS_DIR not set; skipping");
        return;
    };
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/text-rot6.jpg"))
        .expect("fixture");
    let uri = format!("data:image/jpeg;base64,{}", STANDARD.encode(bytes));
    let (status, v) = post(
        app,
        None,
        serde_json::json!({"document": {"type": "image_url", "image_url": uri}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["pages"][0]["orientation_corrected"], false);
    assert_eq!(
        v["pages"][0]["dimensions"]["width"], 160,
        "{}",
        v["pages"][0]
    );
}

#[tokio::test]
async fn pages_carry_blocks() {
    let Some(app) = app(None) else {
        eprintln!("OCR_MODELS_DIR not set; skipping");
        return;
    };
    let (status, v) = post(
        app,
        None,
        serde_json::json!({"document": {"type": "image_url", "image_url": data_uri(FIXTURE)}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let blocks = v["pages"][0]["blocks"].as_array().expect("blocks");
    assert!(!blocks.is_empty(), "{v}");
    assert_eq!(blocks[0]["layout_type"], "Text");
    assert!(blocks[0]["text"].as_str().unwrap().len() > 3);
}
