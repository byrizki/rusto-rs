use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use ndarray::ArrayD;
use rusto::{
    DetectTextResult, EngineError, EngineSession, ImageSource, InferenceSession,
    InitializeConfig, OcrRunOptions, RustO, TextDetector, TextRecognizer,
};

/// A custom inference session wrapper that counts how many inferences were executed.
struct CountingSession {
    inner: EngineSession,
    inference_count: Arc<AtomicUsize>,
}

impl CountingSession {
    fn new(inner: EngineSession, counter: Arc<AtomicUsize>) -> Self {
        Self {
            inner,
            inference_count: counter,
        }
    }
}

impl InferenceSession for CountingSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.inference_count.fetch_add(1, Ordering::SeqCst);
        self.inner.run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.inference_count.fetch_add(1, Ordering::SeqCst);
        self.inner.run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.inference_count.fetch_add(1, Ordering::SeqCst);
        self.inner.run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.inner.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.inner.have_key(key)
    }
}

#[test]
fn test_custom_inference_session_pluggability() {
    let det_path = "models/ppocrv6-tiny/det.rten";
    let rec_path = "models/ppocrv6-tiny/rec.rten";
    let dict_path = "models/ppocrv6-tiny/dict.txt";

    if !std::path::Path::new(det_path).exists() {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path);

    let raw_det_session = EngineSession::from_det_config(&config.det).expect("Load det");
    let raw_rec_session = EngineSession::from_rec_config(&config.rec).expect("Load rec");

    let det_counter = Arc::new(AtomicUsize::new(0));
    let rec_counter = Arc::new(AtomicUsize::new(0));

    let custom_det = CountingSession::new(raw_det_session, Arc::clone(&det_counter));
    let custom_rec = CountingSession::new(raw_rec_session, Arc::clone(&rec_counter));

    let mut ocr = RustO::with_custom_engines(config, custom_det, custom_rec)
        .expect("Initialize with custom engines");

    let img_path = "tests/test_imgs/test_simple.png";
    let res = ocr.detect_text(&ImageSource::Path(img_path.into()), &OcrRunOptions::default())
        .expect("Run OCR with custom engines");

    match res {
        DetectTextResult::Structured(lines) => {
            assert!(!lines.is_empty(), "Should detect at least 1 line");
        }
        _ => panic!("Expected structured output"),
    }

    assert!(det_counter.load(Ordering::SeqCst) >= 1, "Det session was invoked");
    assert!(rec_counter.load(Ordering::SeqCst) >= 1, "Rec session was invoked");
}

#[test]
fn test_with_components_composition() {
    let det_path = "models/ppocrv6-tiny/det.rten";
    let rec_path = "models/ppocrv6-tiny/rec.rten";
    let dict_path = "models/ppocrv6-tiny/dict.txt";

    if !std::path::Path::new(det_path).exists() {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path);
    let det = TextDetector::new(config.det.clone()).expect("Create detector");
    let rec = TextRecognizer::new(config.rec.clone()).expect("Create recognizer");

    let mut ocr = RustO::with_components(config.global, det, rec)
        .expect("Initialize with components");

    let img_path = "tests/test_imgs/test_simple.png";
    let res = ocr.detect_text(&ImageSource::Path(img_path.into()), &OcrRunOptions::default())
        .expect("Run OCR with composed components");

    if let DetectTextResult::Structured(lines) = res {
        assert!(!lines.is_empty());
    } else {
        panic!("Expected structured output");
    }
}
