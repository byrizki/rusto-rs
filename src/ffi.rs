// FFI bindings for C/C++/C#
#![allow(clippy::missing_safety_doc, clippy::needless_borrow)]

use crate::{DetectTextResult, ImageSource, OcrRunOptions, OutputGranularity, RustO, InitializeConfig};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::slice;

const ROCR_INVALID_OPTIONS: c_int = -4;

unsafe fn parse_run_options(options_json: *const c_char) -> Result<OcrRunOptions, c_int> {
    if options_json.is_null() {
        return Err(-1);
    }
    let options = CStr::from_ptr(options_json)
        .to_str()
        .ok()
        .and_then(|json| serde_json::from_str::<OcrRunOptions>(json).ok())
        .ok_or(ROCR_INVALID_OPTIONS)?;
    options.validate().map_err(|_| ROCR_INVALID_OPTIONS)?;
    Ok(options)
}

/// Opaque handle to RustO instance
pub struct ROCRHandle {
    inner: RustO,
}

/// C-compatible text result structure
#[repr(C)]
pub struct CTextResult {
    pub text: *mut c_char,
    pub score: c_float,
    pub box_x1: c_float,
    pub box_y1: c_float,
    pub box_x2: c_float,
    pub box_y2: c_float,
    pub box_x3: c_float,
    pub box_y3: c_float,
    pub box_x4: c_float,
    pub box_y4: c_float,
    pub frame_width: c_float,
    pub frame_height: c_float,
    pub frame_top: c_float,
    pub frame_left: c_float,
}

/// Create a new RustO instance with JSON configuration
///
/// # Safety
/// config_json must be a valid null-terminated UTF-8 JSON string
#[no_mangle]
pub unsafe extern "C" fn rocr_initialize(config_json: *const c_char) -> *mut ROCRHandle {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    let json_str = match CStr::from_ptr(config_json).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let config = match InitializeConfig::from_json(json_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };

    match RustO::initialize(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Run OCR on an image file with runtime output options.
#[no_mangle]
pub unsafe extern "C" fn rocr_detect_text_file(
    handle: *mut ROCRHandle,
    image_path: *const c_char,
    options_json: *const c_char,
    results_out: *mut *mut CTextResult,
    count_out: *mut usize,
) -> c_int {
    if handle.is_null()
        || image_path.is_null()
        || options_json.is_null()
        || results_out.is_null()
        || count_out.is_null()
    {
        return -1;
    }
    let path = match CStr::from_ptr(image_path).to_str() {
        Ok(value) => value,
        Err(_) => return -2,
    };
    let options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(code) => return code,
    };
    if options.output == OutputGranularity::Spatial {
        return -5;
    }
    let detected = match (*handle)
        .inner
        .detect_text(&ImageSource::Path(path.into()), &options)
    {
        Ok(value) => value,
        Err(_) => return -3,
    };
    let DetectTextResult::Structured(projected) = detected else {
        return -5;
    };
    let c_results = text_results_to_c(&projected);
    *count_out = c_results.len();
    *results_out = Box::into_raw(c_results) as *mut CTextResult;
    0
}

/// Run OCR on an image file and return spatial text with runtime options.
#[no_mangle]
pub unsafe extern "C" fn rocr_detect_text_file_spatial(
    handle: *mut ROCRHandle,
    image_path: *const c_char,
    options_json: *const c_char,
) -> *mut c_char {
    if handle.is_null() || image_path.is_null() || options_json.is_null() {
        return std::ptr::null_mut();
    }
    let path = match CStr::from_ptr(image_path).to_str() {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    let mut options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    options.output = OutputGranularity::Spatial;
    let detected = match (*handle)
        .inner
        .detect_text(&ImageSource::Path(path.into()), &options)
    {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    let DetectTextResult::Spatial(text) = detected else {
        return std::ptr::null_mut();
    };
    CString::new(text).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Run OCR on image data with runtime output options.
#[no_mangle]
pub unsafe extern "C" fn rocr_detect_text_data(
    handle: *mut ROCRHandle,
    image_data: *const u8,
    image_len: usize,
    options_json: *const c_char,
    results_out: *mut *mut CTextResult,
    count_out: *mut usize,
) -> c_int {
    if handle.is_null()
        || image_data.is_null()
        || options_json.is_null()
        || results_out.is_null()
        || count_out.is_null()
    {
        return -1;
    }
    let options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(code) => return code,
    };
    if options.output == OutputGranularity::Spatial {
        return -5;
    }
    let detected = match (*handle).inner.detect_text(
        &ImageSource::Bytes(slice::from_raw_parts(image_data, image_len).to_vec()),
        &options,
    ) {
        Ok(value) => value,
        Err(_) => return -3,
    };
    let DetectTextResult::Structured(results) = detected else {
        return -5;
    };
    let c_results = text_results_to_c(&results);
    *count_out = c_results.len();
    *results_out = Box::into_raw(c_results) as *mut CTextResult;
    0
}

/// Run OCR on image data and return spatial text with runtime options.
#[no_mangle]
pub unsafe extern "C" fn rocr_detect_text_data_spatial(
    handle: *mut ROCRHandle,
    image_data: *const u8,
    image_len: usize,
    options_json: *const c_char,
) -> *mut c_char {
    if handle.is_null() || image_data.is_null() || options_json.is_null() {
        return std::ptr::null_mut();
    }
    let mut options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    options.output = OutputGranularity::Spatial;
    let detected = match (*handle).inner.detect_text(
        &ImageSource::Bytes(slice::from_raw_parts(image_data, image_len).to_vec()),
        &options,
    ) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    let DetectTextResult::Spatial(text) = detected else {
        return std::ptr::null_mut();
    };
    CString::new(text).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Free string returned from export functions
///
/// # Safety
/// - s must be a pointer returned from rocr_detect_text_*_spatial
#[no_mangle]
pub unsafe extern "C" fn rocr_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free results returned from structured text detection.
///
/// # Safety
/// - results must be a pointer returned from rocr_detect_text_file or rocr_detect_text_data
/// - count must match the count returned by text detection
#[no_mangle]
pub unsafe extern "C" fn rocr_free_results(results: *mut CTextResult, count: usize) {
    if results.is_null() || count == 0 {
        return;
    }

    let slice = std::slice::from_raw_parts_mut(results, count);
    let boxed = Box::from_raw(slice);
    for result in boxed.iter() {
        if !result.text.is_null() {
            drop(CString::from_raw(result.text));
        }
    }
    drop(boxed);
}

/// Free a RustO instance
///
/// # Safety
/// handle must be a valid pointer returned from rocr_initialize
#[no_mangle]
pub unsafe extern "C" fn rocr_free(handle: *mut ROCRHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get library version
#[no_mangle]
pub extern "C" fn rocr_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

fn text_results_to_c(results: &[crate::TextResult]) -> Box<[CTextResult]> {
    results
        .iter()
        .map(|result| CTextResult {
            text: CString::new(result.text.clone())
                .unwrap_or_default()
                .into_raw(),
            score: result.score,
            box_x1: result.box_points[0].0,
            box_y1: result.box_points[0].1,
            box_x2: result.box_points[1].0,
            box_y2: result.box_points[1].1,
            box_x3: result.box_points[2].0,
            box_y3: result.box_points[2].1,
            box_x4: result.box_points[3].0,
            box_y4: result.box_points[3].1,
            frame_width: result.frame.width,
            frame_height: result.frame.height,
            frame_top: result.frame.top,
            frame_left: result.frame.left,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}
