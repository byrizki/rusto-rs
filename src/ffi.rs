// FFI bindings for C/C++/C#
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::slice;
use crate::{RustO, RustOConfig, TextResult};

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
}

/// Create a new RustO instance
///
/// # Safety
/// All string pointers must be valid null-terminated UTF-8 strings
#[no_mangle]
pub unsafe extern "C" fn rocr_new(
    det_model_path: *const c_char,
    rec_model_path: *const c_char,
    dict_path: *const c_char
) -> *mut ROCRHandle {
    if det_model_path.is_null() || rec_model_path.is_null() || dict_path.is_null() {
        return std::ptr::null_mut();
    }

    let det_model = match CStr::from_ptr(det_model_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };

    let rec_model = match CStr::from_ptr(rec_model_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };

    let dict = match CStr::from_ptr(dict_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };

    let config = RustOConfig {
        det_model_path: det_model,
        rec_model_path: rec_model,
        dict_path: dict,
        ..Default::default()
    };

    match RustO::new(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Run OCR on an image file
///
/// # Safety
/// - handle must be a valid pointer returned from rocr_new
/// - image_path must be a valid null-terminated UTF-8 string
/// - results_out will be allocated and must be freed with rocr_free_results
#[no_mangle]
pub unsafe extern "C" fn rocr_ocr_file(
    handle: *mut ROCRHandle,
    image_path: *const c_char,
    results_out: *mut *mut CTextResult,
    count_out: *mut usize,
) -> c_int {
    if handle.is_null() || image_path.is_null() || results_out.is_null() || count_out.is_null() {
        return -1;
    }

    let ocr = &mut (*handle).inner;

    let path = match CStr::from_ptr(image_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let results = match ocr.ocr(path) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(results);
    *count_out = c_results.len();
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);

    0
}

/// Run OCR on image data in memory
///
/// # Safety
/// - handle must be a valid pointer returned from rocr_new
/// - image_data must point to valid image bytes of length image_len
/// - results_out will be allocated and must be freed with rocr_free_results
#[no_mangle]
pub unsafe extern "C" fn rocr_ocr_data(
    handle: *mut ROCRHandle,
    image_data: *const u8,
    image_len: usize,
    results_out: *mut *mut CTextResult,
    count_out: *mut usize,
) -> c_int {
    if handle.is_null() || image_data.is_null() || results_out.is_null() || count_out.is_null() {
        return -1;
    }

    let ocr = &mut (*handle).inner;
    let data = slice::from_raw_parts(image_data, image_len);

    let results = match ocr.ocr_from_bytes(data) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(results);
    *count_out = c_results.len();
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);

    0
}

/// Free results returned from rocr_ocr
///
/// # Safety
/// - results must be a pointer returned from rocr_ocr_file or rocr_ocr_data
/// - count must match the count returned from rocr_ocr
#[no_mangle]
pub unsafe extern "C" fn rocr_free_results(results: *mut CTextResult, count: usize) {
    if results.is_null() {
        return;
    }

    let results_vec = Vec::from_raw_parts(results, count, count);
    for result in results_vec {
        if !result.text.is_null() {
            drop(CString::from_raw(result.text));
        }
    }
}

/// Free a RustO instance
///
/// # Safety
/// handle must be a valid pointer returned from rocr_new
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

// Helper function to convert Rust results to C results
fn results_to_c(results: Vec<TextResult>) -> Vec<CTextResult> {
    results
        .into_iter()
        .map(|r| {
            let text = CString::new(r.text).unwrap();
            CTextResult {
                text: text.into_raw(),
                score: r.score,
                box_x1: r.box_points[0].0,
                box_y1: r.box_points[0].1,
                box_x2: r.box_points[1].0,
                box_y2: r.box_points[1].1,
                box_x3: r.box_points[2].0,
                box_y3: r.box_points[2].1,
                box_x4: r.box_points[3].0,
                box_y4: r.box_points[3].1,
            }
        })
        .collect()
}
