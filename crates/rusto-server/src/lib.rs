//! Mistral-compatible `POST /v1/ocr` backed by rusto-rs (PP-OCRv6 models).
//!
//! `pages[].markdown` is the recognized lines in the order rusto returns them. Each page also carries
//! non-Mistral `lines` (text, score, box) and `blocks` (rusto's layout regions with their member text)
//! that Mistral clients ignore. No tables, images or PDFs yet.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rusto::{
    cluster_text_results_into_blocks, DetectTextResult, ImageSource, InitializeConfig,
    OcrRunOptions, RustO,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Mutex, OnceCell, Semaphore};
const MAX_DOC_BYTES: usize = 32 * 1024 * 1024;

pub struct Config {
    /// Named PP-OCRv6 model directories (each holds det.onnx, rec.onnx, dict.txt), e.g. ("tiny", ...).
    pub models: Vec<(String, PathBuf)>,
    /// Model used when the request names none, or names a Mistral model such as `mistral-ocr-latest`.
    pub default_model: String,
    /// Engines per model. An engine is single-threaded; models load on first use.
    pub pool_size: usize,
    pub api_key: Option<String>,
    /// Hosts `document_url` / `image_url` may be fetched from. Empty = no remote fetch.
    pub allowed_hosts: Vec<String>,
    /// Apply the EXIF orientation before OCR when a request does not say. Off by default: EXIF tags are
    /// not always accurate, so the server never rotates unless it is asked to.
    pub autorotate: bool,
}

/// One model's engines, loaded on first request so unused models cost no memory.
struct ModelPool {
    dir: PathBuf,
    size: usize,
    pool: OnceCell<Pool>,
}

struct Pool {
    engines: Mutex<Vec<RustO>>,
    permits: Semaphore,
}

pub struct AppState {
    models: BTreeMap<String, ModelPool>,
    default_model: String,
    api_key: Option<String>,
    allowed_hosts: Vec<String>,
    autorotate: bool,
}

pub fn build_state(cfg: Config) -> Result<Arc<AppState>, String> {
    let size = cfg.pool_size.clamp(1, 64);
    let mut models = BTreeMap::new();
    for (name, dir) in cfg.models {
        for file in ["det.onnx", "rec.onnx", "dict.txt"] {
            if !dir.join(file).is_file() {
                return Err(format!(
                    "model '{name}': missing {}",
                    dir.join(file).display()
                ));
            }
        }
        models.insert(
            name,
            ModelPool {
                dir,
                size,
                pool: OnceCell::new(),
            },
        );
    }
    if !models.contains_key(&cfg.default_model) {
        return Err(format!(
            "default model '{}' not found; have: {}",
            cfg.default_model,
            models.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(Arc::new(AppState {
        models,
        default_model: cfg.default_model,
        api_key: cfg.api_key,
        allowed_hosts: cfg.allowed_hosts,
        autorotate: cfg.autorotate,
    }))
}

impl ModelPool {
    async fn pool(&self) -> Result<&Pool, String> {
        self.pool
            .get_or_try_init(|| async {
                let dir = self.dir.clone();
                let size = self.size;
                tokio::task::spawn_blocking(move || {
                    let mut engines = Vec::with_capacity(size);
                    for _ in 0..size {
                        engines.push(
                            RustO::initialize(InitializeConfig::ppv6(
                                dir.join("det.onnx"),
                                dir.join("rec.onnx"),
                                dir.join("dict.txt"),
                            ))
                            .map_err(|e| format!("rusto init failed: {e}"))?,
                        );
                    }
                    Ok::<_, String>(Pool {
                        engines: Mutex::new(engines),
                        permits: Semaphore::new(size),
                    })
                })
                .await
                .map_err(|e| e.to_string())?
            })
            .await
    }
}

impl AppState {
    /// `rusto-ppocrv6-<name>` and bare `<name>` select a model; Mistral model ids and no
    /// model select the default; anything else is an error rather than a silent fallback.
    fn resolve(&self, requested: Option<&str>) -> Result<&str, ApiError> {
        let Some(m) = requested.filter(|m| !m.is_empty()) else {
            return Ok(&self.default_model);
        };
        let name = m.strip_prefix("rusto-ppocrv6-").unwrap_or(m);
        if let Some((key, _)) = self.models.get_key_value(name) {
            return Ok(key);
        }
        if m.starts_with("mistral-ocr") {
            return Ok(&self.default_model);
        }
        Err(bad_request(format!(
            "unknown model '{m}'; available: {}",
            self.models.keys().cloned().collect::<Vec<_>>().join(", ")
        )))
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/ocr", post(ocr))
        .layer(DefaultBodyLimit::max(MAX_DOC_BYTES * 2))
        .with_state(state)
}

// ---- request / response ----------------------------------------------------

#[derive(Deserialize)]
struct OcrRequest {
    model: Option<String>,
    document: Document,
    /// Not part of the Mistral schema. Apply the EXIF orientation before OCR; when absent the server's
    /// `autorotate` setting (default off) applies. EXIF tags are not always accurate.
    autorotate: Option<bool>,
}

#[derive(Deserialize)]
struct Document {
    #[serde(rename = "type")]
    kind: String,
    document_url: Option<UrlField>,
    image_url: Option<UrlField>,
}

/// Mistral accepts the URL either as a bare string or as `{"url": "..."}`.
#[derive(Deserialize)]
#[serde(untagged)]
enum UrlField {
    Plain(String),
    Object { url: String },
}

impl UrlField {
    fn url(&self) -> &str {
        match self {
            UrlField::Plain(u) | UrlField::Object { url: u } => u,
        }
    }
}

#[derive(Serialize)]
struct OcrResponse {
    pages: Vec<Page>,
    model: String,
    usage_info: UsageInfo,
}

#[derive(Serialize)]
struct Page {
    index: u32,
    markdown: String,
    images: Vec<serde_json::Value>,
    dimensions: Dimensions,
    /// Not part of the Mistral schema; ignored by Mistral clients.
    lines: Vec<Line>,
    /// Not part of the Mistral schema. rusto's layout regions in reading order, each with its member text.
    /// Without a layout model every line is its own block.
    blocks: Vec<Block>,
    /// Not part of the Mistral schema. True when an EXIF orientation was applied to the pixels before OCR.
    orientation_corrected: bool,
    /// Not part of the Mistral schema. Dominant slope of the lines in degrees (0 = horizontal, clockwise
    /// positive, image y points down): length-weighted median of each box's top edge. Null with no lines.
    text_angle: Option<f32>,
}

#[derive(Serialize)]
struct Dimensions {
    dpi: Option<u32>,
    height: u32,
    width: u32,
}

#[derive(Serialize)]
struct Block {
    region_id: i32,
    /// Text, Title, Figure, FigureCaption, Table, TableCaption, Header, Footer, Reference or Equation.
    layout_type: String,
    text: String,
    score: f32,
    #[serde(rename = "box")]
    corners: [(f32, f32); 4],
}

#[derive(Serialize)]
struct Line {
    text: String,
    score: f32,
    /// Four corners: top-left, top-right, bottom-right, bottom-left.
    #[serde(rename = "box")]
    corners: [(f32, f32); 4],
}

#[derive(Serialize)]
struct UsageInfo {
    pages_processed: u32,
    doc_size_bytes: usize,
}

// ---- errors (Mistral error envelope) ---------------------------------------

struct ApiError(StatusCode, &'static str, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "object": "error",
            "message": self.2,
            "type": self.1,
            "param": null,
            "code": self.0.as_u16().to_string(),
        });
        (self.0, Json(body)).into_response()
    }
}

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, "invalid_request_error", msg.into())
}

// ---- handler -----------------------------------------------------------------

async fn ocr(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<OcrRequest>,
) -> Result<Json<OcrResponse>, ApiError> {
    authorize(&state, &headers)?;

    let url = match req.document.kind.as_str() {
        "image_url" => req.document.image_url.as_ref(),
        "document_url" => req.document.document_url.as_ref(),
        other => return Err(bad_request(format!("unsupported document type: {other}"))),
    }
    .ok_or_else(|| bad_request("document is missing its url field"))?
    .url()
    .to_owned();

    let model = state.resolve(req.model.as_deref())?.to_owned();
    let bytes = load_document(&state, &url).await?;
    if bytes.starts_with(b"%PDF") {
        return Err(bad_request("PDF input is not supported yet; send an image"));
    }
    let doc_size = bytes.len();
    // Phones store the sensor's pixels plus an EXIF orientation; rusto ignores the tag, so apply it first.
    let autorotate = req.autorotate.unwrap_or(state.autorotate);
    let (bytes, orientation_corrected) = tokio::task::spawn_blocking(move || {
        if autorotate {
            upright(bytes)
        } else {
            (bytes, false)
        }
    })
    .await
    .map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            e.to_string(),
        )
    })?;
    let size = imagesize::blob_size(&bytes)
        .map_err(|_| bad_request("document is not a recognizable image"))?;

    let (lines, blocks) = run_ocr(&state, &model, bytes).await?;
    let text_angle = text_angle(&lines);
    let markdown = lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(Json(OcrResponse {
        pages: vec![Page {
            index: 0,
            markdown,
            images: vec![],
            dimensions: Dimensions {
                dpi: None,
                height: size.height as u32,
                width: size.width as u32,
            },
            lines,
            blocks,
            orientation_corrected,
            text_angle,
        }],
        model: format!("rusto-ppocrv6-{model}"),
        usage_info: UsageInfo {
            pages_processed: 1,
            doc_size_bytes: doc_size,
        },
    }))
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &state.api_key else {
        return Ok(());
    };
    let ok = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|got| got == key);
    if ok {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key".into(),
        ))
    }
}

async fn load_document(state: &AppState, url: &str) -> Result<Vec<u8>, ApiError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest
            .split_once(',')
            .ok_or_else(|| bad_request("malformed data URI"))?;
        if !meta.ends_with(";base64") {
            return Err(bad_request("data URI must be base64-encoded"));
        }
        let bytes = STANDARD
            .decode(data.trim())
            .map_err(|_| bad_request("invalid base64 in data URI"))?;
        return check_size(bytes);
    }

    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err(bad_request(
            "document url must be a data: URI or http(s) URL",
        ));
    }
    let host = url
        .split("://")
        .nth(1)
        .and_then(|r| r.split(['/', '?', '#']).next())
        .and_then(|authority| authority.rsplit('@').next())
        .map(|h| h.split(':').next().unwrap_or(h).to_ascii_lowercase())
        .unwrap_or_default();
    if !state
        .allowed_hosts
        .iter()
        .any(|h| h.eq_ignore_ascii_case(&host))
    {
        return Err(bad_request(format!(
            "fetching from host '{host}' is not allowed; send a data: URI or add it to OCR_ALLOWED_HOSTS"
        )));
    }

    let url = url.to_owned();
    let fetched = tokio::task::spawn_blocking(move || {
        use std::io::Read;
        // No redirects: a redirect could leave the allowlisted host.
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout(std::time::Duration::from_secs(30))
            .build();
        let resp = agent.get(&url).call().map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        resp.into_reader()
            .take(MAX_DOC_BYTES as u64 + 1)
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(buf)
    })
    .await
    .map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            e.to_string(),
        )
    })?
    .map_err(|e| bad_request(format!("could not fetch document: {e}")))?;
    check_size(fetched)
}

fn check_size(bytes: Vec<u8>) -> Result<Vec<u8>, ApiError> {
    if bytes.is_empty() {
        return Err(bad_request("document is empty"));
    }
    if bytes.len() > MAX_DOC_BYTES {
        return Err(bad_request("document exceeds the 32 MB limit"));
    }
    Ok(bytes)
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let data: Vec<_> = state
        .models
        .keys()
        .map(|name| json!({"id": name, "object": "model", "default": *name == state.default_model}))
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn run_ocr(
    state: &Arc<AppState>,
    model: &str,
    bytes: Vec<u8>,
) -> Result<(Vec<Line>, Vec<Block>), ApiError> {
    let server_error = |m: String| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "server_error", m);

    let pool = state.models[model].pool().await.map_err(&server_error)?;
    let _permit = pool
        .permits
        .acquire()
        .await
        .map_err(|e| server_error(e.to_string()))?;
    let mut engine = pool
        .engines
        .lock()
        .await
        .pop()
        .ok_or_else(|| server_error("engine pool empty".into()))?;

    // Inference is CPU-bound and cannot be interrupted; run it off the async threads
    // and always return the engine to the pool.
    let (engine, result) = tokio::task::spawn_blocking(move || {
        let r = engine.detect_text(&ImageSource::Bytes(bytes), &OcrRunOptions::default());
        (engine, r)
    })
    .await
    .map_err(|e| server_error(e.to_string()))?;
    pool.engines.lock().await.push(engine);

    match result {
        Ok(DetectTextResult::Structured(rs)) => {
            let blocks = cluster_text_results_into_blocks(&rs)
                .into_iter()
                .map(|b| Block {
                    region_id: b.region_id,
                    layout_type: format!("{:?}", b.layout_type),
                    text: b.text(),
                    score: b.score,
                    corners: b.box_points,
                })
                .collect();
            let lines = rs
                .into_iter()
                .map(|r| Line {
                    text: r.text,
                    score: r.score,
                    corners: r.box_points,
                })
                .collect();
            Ok((lines, blocks))
        }
        Ok(DetectTextResult::Spatial(_)) => Err(server_error("unexpected spatial output".into())),
        Err(e) => Err(bad_request(format!("OCR failed: {e}"))),
    }
}

/// Applies the EXIF orientation to the pixels. Returns the bytes unchanged (and false) when there is
/// nothing to apply or the image cannot be decoded here; rotated images are re-encoded as JPEG (q95).
fn upright(bytes: Vec<u8>) -> (Vec<u8>, bool) {
    match rotated_copy(&bytes) {
        Some(out) => (out, true),
        None => (bytes, false),
    }
}

fn rotated_copy(bytes: &[u8]) -> Option<Vec<u8>> {
    use image::{metadata::Orientation, DynamicImage, ImageDecoder, ImageReader};
    use std::io::Cursor;

    let mut decoder = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    if orientation == Orientation::NoTransforms {
        return None;
    }
    let mut img = DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95);
    img.to_rgb8().write_with_encoder(encoder).ok()?;
    Some(out)
}

/// Length-weighted median of each line's top-edge angle, in degrees.
fn text_angle(lines: &[Line]) -> Option<f32> {
    let mut v: Vec<(f32, f32)> = lines
        .iter()
        .filter_map(|l| {
            let (dx, dy) = (
                l.corners[1].0 - l.corners[0].0,
                l.corners[1].1 - l.corners[0].1,
            );
            let len = dx.hypot(dy);
            (len > 0.0).then(|| (dy.atan2(dx).to_degrees(), len))
        })
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = v.iter().map(|x| x.1).sum::<f32>() / 2.0;
    let mut acc = 0.0;
    for (angle, w) in &v {
        acc += w;
        if acc >= half {
            return Some((angle * 10.0).round() / 10.0);
        }
    }
    v.last().map(|x| x.0)
}
