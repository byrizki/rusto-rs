use napi::bindgen_prelude::*;
use napi_derive::napi;
use rusto_rs::{DetectTextResult, ImageSource, OcrRunOptions, RustO};

#[napi(js_name = "RustO")]
pub struct JsRustO {
    inner: RustO,
}

#[napi]
impl JsRustO {
    #[napi(factory)]
    pub fn initialize(config_json: String) -> Result<Self> {
        rusto_rs::ensure_rten_registered();
        let config = rusto_rs::download::parse_config_with_auto_download(&config_json)
            .map_err(|e| Error::new(Status::InvalidArg, format!("{e}")))?;

        let inner = RustO::initialize(config)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to initialize RustO: {e}")))?;

        Ok(Self { inner })
    }

    #[napi]
    pub fn download_model(
        model: Option<String>,
        lang: Option<String>,
        cache_dir: Option<String>,
    ) -> Result<Vec<String>> {
        let cache_path = rusto_rs::models::get_cache_dir(cache_dir.as_deref().map(std::path::Path::new));
        let downloaded = rusto_rs::download::download_model_pipeline(
            model.as_deref(),
            lang.as_deref(),
            false,
            false,
            false,
            None,
            false,
            &cache_path,
        ).map_err(|e| Error::new(Status::GenericFailure, format!("Failed to download model: {e}")))?;
        Ok(downloaded.into_iter().map(|p| p.to_string_lossy().to_string()).collect())
    }

    #[napi]
    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    #[napi]
    pub fn detect_text_file(&mut self, path: String, options_json: Option<String>) -> Result<String> {
        let options = match options_json {
            Some(json) if !json.trim().is_empty() => <OcrRunOptions as nanoserde::DeJson>::deserialize_json(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };

        let result = self.inner.detect_text(&ImageSource::Path(path.into()), &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Detection failed: {e}")))?;

        match result {
            DetectTextResult::Structured(items) => Ok(nanoserde::SerJson::serialize_json(&items)),
            DetectTextResult::Spatial(text) => Ok(nanoserde::SerJson::serialize_json(&text)),
        }
    }

    #[napi]
    pub fn detect_text_data(&mut self, data: Buffer, options_json: Option<String>) -> Result<String> {
        let options = match options_json {
            Some(json) if !json.trim().is_empty() => <OcrRunOptions as nanoserde::DeJson>::deserialize_json(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };

        let bytes: Vec<u8> = data.as_ref().to_vec();
        let result = self.inner.detect_text(&ImageSource::Bytes(bytes), &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Detection failed: {e}")))?;

        match result {
            DetectTextResult::Structured(items) => Ok(nanoserde::SerJson::serialize_json(&items)),
            DetectTextResult::Spatial(text) => Ok(nanoserde::SerJson::serialize_json(&text)),
        }
    }

    #[napi]
    pub fn detect_text_spatial(&mut self, path: String, options_json: Option<String>) -> Result<String> {
        let mut options = match options_json {
            Some(json) if !json.trim().is_empty() => <OcrRunOptions as nanoserde::DeJson>::deserialize_json(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };
        options.output = rusto_rs::OutputGranularity::Spatial;

        let result = self.inner.detect_text(&ImageSource::Path(path.into()), &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Spatial detection failed: {e}")))?;

        match result {
            DetectTextResult::Spatial(s) => Ok(s),
            _ => Err(Error::new(Status::GenericFailure, "Expected spatial output")),
        }
    }

    #[napi]
    pub fn detect_text_rgb(
        &mut self,
        pixels: Buffer,
        width: u32,
        height: u32,
        stride: Option<u32>,
        options_json: Option<String>,
    ) -> Result<String> {
        let options = match options_json {
            Some(json) if !json.trim().is_empty() => <OcrRunOptions as nanoserde::DeJson>::deserialize_json(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };

        let effective_stride = stride.unwrap_or(width.saturating_mul(3)) as usize;
        let result = self
            .inner
            .detect_text_rgb(pixels.as_ref(), width, height, effective_stride, &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("RGB detection failed: {e}")))?;

        match result {
            DetectTextResult::Structured(items) => Ok(nanoserde::SerJson::serialize_json(&items)),
            DetectTextResult::Spatial(text) => Ok(nanoserde::SerJson::serialize_json(&text)),
        }
    }

    #[napi]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

#[napi]
pub fn rusto_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

