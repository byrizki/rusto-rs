#![cfg(all(feature = "ffi", not(feature = "use-opencv")))]
use rusto::{ffi::*, DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::{ffi::CString, path::PathBuf, ptr};

#[test]
fn null_rgb_handle_resets_outputs() {
    let mut results = ptr::dangling_mut();
    let mut count = 99;
    let code = unsafe {
        rocr_detect_text_rgb(
            ptr::null_mut(),
            ptr::null(),
            0,
            0,
            0,
            0,
            ptr::null(),
            &mut results,
            &mut count,
        )
    };
    assert_eq!(code, -1);
    assert!(results.is_null());
    assert_eq!(count, 0);
}

/// Run with RUSTO_TEST_MODELS pointing to PP-OCRv6 det.onnx/rec.onnx/dict.txt.
#[test]
#[ignore = "requires downloaded PP-OCRv6 models; set RUSTO_TEST_MODELS"]
fn rgb_matches_png_and_checks_ffi_contract() {
    let dir = PathBuf::from(std::env::var("RUSTO_TEST_MODELS").expect("RUSTO_TEST_MODELS"));
    let config = InitializeConfig::ppv6(
        dir.join("det.onnx"),
        dir.join("rec.onnx"),
        dir.join("dict.txt"),
    );
    let mut ocr = RustO::initialize(config.clone()).unwrap();
    let png = include_bytes!("fixtures/rgb-ocr.png");
    let img = image::load_from_memory(png).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let row = w as usize * 3;
    let stride = row + 13;
    let mut padded = vec![17; stride * (h as usize - 1) + row];
    for y in 0..h as usize {
        padded[y * stride..y * stride + row].copy_from_slice(&img.as_raw()[y * row..y * row + row]);
    }
    let original = padded.clone();
    let opts = OcrRunOptions::default();
    let expected = ocr
        .detect_text(&ImageSource::Bytes(png.to_vec()), &opts)
        .unwrap();
    let DetectTextResult::Structured(ref lines) = expected else {
        panic!()
    };
    assert!(lines.iter().any(|line| line.text.contains("HELLO")));
    assert_eq!(
        ocr.detect_text_rgb(&padded, w, h, stride, &opts).unwrap(),
        expected
    );
    assert_eq!(padded, original);
    assert!(ocr
        .detect_text_rgb(&padded[..padded.len() - 1], w, h, stride, &opts)
        .is_err());
    let config_json = CString::new(serde_json::to_string(&config).unwrap()).unwrap();
    let options = CString::new("{}").unwrap();
    let handle = unsafe { rocr_initialize(config_json.as_ptr()) };
    assert!(!handle.is_null());
    unsafe {
        for (len, width, height, s, option, code) in [
            (padded.len() - 1, w, h, stride, options.clone(), -2),
            (padded.len(), 0, h, stride, options.clone(), -2),
            (padded.len(), w, h, row - 1, options.clone(), -2),
            (padded.len(), w, h, usize::MAX, options.clone(), -2),
            (
                padded.len(),
                w,
                h,
                stride,
                CString::new("invalid").unwrap(),
                -4,
            ),
            (
                padded.len(),
                w,
                h,
                stride,
                CString::new("{\"output\":\"spatial\"}").unwrap(),
                -5,
            ),
        ] {
            let mut results = ptr::dangling_mut();
            let mut count = 99;
            assert_eq!(
                rocr_detect_text_rgb(
                    handle,
                    padded.as_ptr(),
                    len,
                    width,
                    height,
                    s,
                    option.as_ptr(),
                    &mut results,
                    &mut count
                ),
                code
            );
            assert!(results.is_null());
            assert_eq!(count, 0);
        }
        for _ in 0..3 {
            let mut results = ptr::null_mut();
            let mut count = 0;
            assert_eq!(
                rocr_detect_text_rgb(
                    handle,
                    padded.as_ptr(),
                    padded.len(),
                    w,
                    h,
                    stride,
                    options.as_ptr(),
                    &mut results,
                    &mut count
                ),
                0
            );
            assert_eq!(count, lines.len());
            for (actual, expected) in std::slice::from_raw_parts(results, count).iter().zip(lines) {
                assert_eq!(
                    std::ffi::CStr::from_ptr(actual.text).to_str().unwrap(),
                    expected.text
                );
                assert_eq!(actual.score, expected.score);
                assert_eq!(actual.box_x1, expected.box_points[0].0);
            }
            rocr_free_results(results, count);
        }
        rocr_free(handle);
    }
}
