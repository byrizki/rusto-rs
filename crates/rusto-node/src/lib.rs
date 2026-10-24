use napi::bindgen_prelude::*;
use napi_derive::napi;
use rusto_rs::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};

#[napi(js_name = "RustO")]
pub struct JsRustO {
    inner: RustO,
}

#[napi]
impl JsRustO {
    #[napi(factory)]
    pub fn initialize(config_json: String) -> Result<Self> {
        let config: InitializeConfig = serde_json::from_str(&config_json)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid config JSON: {e}")))?;

        let inner = RustO::initialize(config)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to initialize RustO: {e}")))?;

        Ok(Self { inner })
    }

    #[napi]
    pub fn detect_text_file(&mut self, path: String, options_json: Option<String>) -> Result<String> {
        let options = match options_json {
            Some(json) if !json.trim().is_empty() => serde_json::from_str::<OcrRunOptions>(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };

        let result = self.inner.detect_text(&ImageSource::Path(path.into()), &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Detection failed: {e}")))?;

        match result {
            DetectTextResult::Structured(items) => serde_json::to_string(&items)
                .map_err(|e| Error::new(Status::GenericFailure, format!("Serialization failed: {e}"))),
            DetectTextResult::Spatial(text) => serde_json::to_string(&text)
                .map_err(|e| Error::new(Status::GenericFailure, format!("Serialization failed: {e}"))),
        }
    }

    #[napi]
    pub fn detect_text_data(&mut self, data: Buffer, options_json: Option<String>) -> Result<String> {
        let options = match options_json {
            Some(json) if !json.trim().is_empty() => serde_json::from_str::<OcrRunOptions>(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid options JSON: {e}")))?,
            _ => OcrRunOptions::default(),
        };

        let bytes: Vec<u8> = data.as_ref().to_vec();
        let result = self.inner.detect_text(&ImageSource::Bytes(bytes), &options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Detection failed: {e}")))?;

        match result {
            DetectTextResult::Structured(items) => serde_json::to_string(&items)
                .map_err(|e| Error::new(Status::GenericFailure, format!("Serialization failed: {e}"))),
            DetectTextResult::Spatial(text) => serde_json::to_string(&text)
                .map_err(|e| Error::new(Status::GenericFailure, format!("Serialization failed: {e}"))),
        }
    }

    #[napi]
    pub fn detect_text_spatial(&mut self, path: String, options_json: Option<String>) -> Result<String> {
        let mut options = match options_json {
            Some(json) if !json.trim().is_empty() => serde_json::from_str::<OcrRunOptions>(&json)
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
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
