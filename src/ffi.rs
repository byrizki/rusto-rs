// FFI bindings for C/C++/C#
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::slice;
use crate::{RustO, RustOConfig, RustOOutput};
use crate::image_impl::Mat;

/// Opaque handle to RustO instance
pub struct ROCRHandle {
    inner: RustO,
}

/// Opaque handle to RustOOutput result
pub struct ROCROutputHandle {
    inner: RustOOutput,
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
pub unsafe extern "C" fn rocr_new_with_config(
    config_json: *const c_char,
) -> *mut ROCRHandle {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    let json_str = match CStr::from_ptr(config_json).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let config = match RustOConfig::from_json(json_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };

    match RustO::new(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })),
        Err(_) => std::ptr::null_mut(),
    }
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

    let config = RustOConfig::new_ppv5(det_model, rec_model, dict);

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

    let results = match ocr.run(path) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(&results);
    *count_out = c_results.len();
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);

    0
}

/// Run OCR on an image file and return output handle for formatting
///
/// # Safety
/// - handle must be a valid pointer returned from rocr_new
/// - image_path must be a valid null-terminated UTF-8 string
/// - output_out will contain a handle that must be freed with rocr_free_output
#[no_mangle]
pub unsafe extern "C" fn rocr_ocr_file_with_output(
    handle: *mut ROCRHandle,
    image_path: *const c_char,
    output_out: *mut *mut ROCROutputHandle,
) -> c_int {
    if handle.is_null() || image_path.is_null() || output_out.is_null() {
        return -1;
    }

    let ocr = &mut (*handle).inner;

    let path = match CStr::from_ptr(image_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let results = match ocr.run(path) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    *output_out = Box::into_raw(Box::new(ROCROutputHandle { inner: results }));
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

    #[cfg(not(feature = "use-opencv"))]
    let img = match image::load_from_memory(data) {
        Ok(dynamic_img) => Mat::new(dynamic_img),
        Err(_) => return -3,
    };

    #[cfg(feature = "use-opencv")]
    let img = {
        // For OpenCV backend, we need to decode manually or use imdecode
        // But since we don't have imdecode exposed in image_impl (only imread), 
        // fallback or error out for now unless we add imdecode.
        // Assuming pure rust backend for now as per user context "Pure Rust OCR".
        return -4; 
    };

    let results = match ocr.run_on_mat(&img) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(&results);
    *count_out = c_results.len();
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);

    0
}

/// Run OCR on image data and return output handle for formatting
///
/// # Safety
/// - handle must be a valid pointer returned from rocr_new
/// - image_data must point to valid image bytes of length image_len
/// - output_out will contain a handle that must be freed with rocr_free_output
#[no_mangle]
pub unsafe extern "C" fn rocr_ocr_data_with_output(
    handle: *mut ROCRHandle,
    image_data: *const u8,
    image_len: usize,
    output_out: *mut *mut ROCROutputHandle,
) -> c_int {
    if handle.is_null() || image_data.is_null() || output_out.is_null() {
        return -1;
    }

    let ocr = &mut (*handle).inner;
    let data = slice::from_raw_parts(image_data, image_len);

    #[cfg(not(feature = "use-opencv"))]
    let img = match image::load_from_memory(data) {
        Ok(dynamic_img) => Mat::new(dynamic_img),
        Err(_) => return -3,
    };

    #[cfg(feature = "use-opencv")]
    let img = {
        return -4;
    };

    let results = match ocr.run_on_mat(&img) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    *output_out = Box::into_raw(Box::new(ROCROutputHandle { inner: results }));
    0
}

/// Export output to raw format
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
/// - returned string must be freed with rocr_free_string
#[no_mangle]
pub unsafe extern "C" fn rocr_output_to_raw(output: *const ROCROutputHandle) -> *mut c_char {
    if output.is_null() {
        return std::ptr::null_mut();
    }

    let output_ref = &(*output).inner;
    let text = output_ref.to_raw();
    
    match CString::new(text) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Export output to CSV format
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
/// - returned string must be freed with rocr_free_string
#[no_mangle]
pub unsafe extern "C" fn rocr_output_to_csv(output: *const ROCROutputHandle) -> *mut c_char {
    if output.is_null() {
        return std::ptr::null_mut();
    }

    let output_ref = &(*output).inner;
    let text = output_ref.to_csv();
    
    match CString::new(text) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Export output to text with position format
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
/// - returned string must be freed with rocr_free_string
#[no_mangle]
pub unsafe extern "C" fn rocr_output_to_text_with_position(
    output: *const ROCROutputHandle
) -> *mut c_char {
    if output.is_null() {
        return std::ptr::null_mut();
    }

    let output_ref = &(*output).inner;
    let text = output_ref.to_text_with_position();
    
    match CString::new(text) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Export output to spatial text format
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
/// - returned string must be freed with rocr_free_string
#[no_mangle]
pub unsafe extern "C" fn rocr_output_to_spatial_text(
    output: *const ROCROutputHandle,
    y_threshold_multiplier: c_float,
    x_threshold_multiplier: c_float,
) -> *mut c_char {
    if output.is_null() {
        return std::ptr::null_mut();
    }

    let output_ref = &(*output).inner;
    
    let y_mult = if y_threshold_multiplier <= 0.0 {
        None
    } else {
        Some(y_threshold_multiplier)
    };
    
    let x_mult = if x_threshold_multiplier <= 0.0 {
        None
    } else {
        Some(x_threshold_multiplier)
    };
    
    let text = output_ref.to_spatial_text(y_mult, x_mult);
    
    match CString::new(text) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get results from output handle as C array
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
/// - results_out will be allocated and must be freed with rocr_free_results
#[no_mangle]
pub unsafe extern "C" fn rocr_output_get_results(
    output: *const ROCROutputHandle,
    results_out: *mut *mut CTextResult,
    count_out: *mut usize,
) -> c_int {
    if output.is_null() || results_out.is_null() || count_out.is_null() {
        return -1;
    }

    let output_ref = &(*output).inner;
    let c_results = results_to_c(output_ref);
    *count_out = c_results.len();
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);

    0
}

/// Free output handle
///
/// # Safety
/// - output must be a valid pointer returned from rocr_ocr_*_with_output
#[no_mangle]
pub unsafe extern "C" fn rocr_free_output(output: *mut ROCROutputHandle) {
    if !output.is_null() {
        drop(Box::from_raw(output));
    }
}

/// Free string returned from export functions
///
/// # Safety
/// - s must be a pointer returned from rocr_output_to_* functions
#[no_mangle]
pub unsafe extern "C" fn rocr_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
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
fn results_to_c(results: &RustOOutput) -> Vec<CTextResult> {
    let mut c_results = Vec::with_capacity(results.boxes.len());
    
    for (i, bbox) in results.boxes.iter().enumerate() {
        if i >= results.txts.len() || i >= results.scores.len() {
            break;
        }
        
        let x1 = bbox[0].x;
        let y1 = bbox[0].y;
        let x2 = bbox[1].x;
        let y2 = bbox[1].y;
        let x3 = bbox[2].x;
        let y3 = bbox[2].y;
        let x4 = bbox[3].x;
        let y4 = bbox[3].y;
        
        let min_x = x1.min(x2).min(x3).min(x4);
        let max_x = x1.max(x2).max(x3).max(x4);
        let min_y = y1.min(y2).min(y3).min(y4);
        let max_y = y1.max(y2).max(y3).max(y4);
        
        let text = CString::new(results.txts[i].clone()).unwrap_or_default();
        
        c_results.push(CTextResult {
            text: text.into_raw(),
            score: results.scores[i],
            box_x1: x1,
            box_y1: y1,
            box_x2: x2,
            box_y2: y2,
            box_x3: x3,
            box_y3: y3,
            box_x4: x4,
            box_y4: y4,
            frame_width: max_x - min_x,
            frame_height: max_y - min_y,
            frame_top: min_y,
            frame_left: min_x,
        });
    }
    
    c_results
}

// ============================================================================
// Android JNI Bindings (com.byrizki.rusto.RustO)
// ============================================================================

#[cfg(feature = "ffi")]
use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray, JLongArray, JObjectArray, JString};
#[cfg(feature = "ffi")]
use jni::sys::{jfloat, jint, jlong, jstring};
#[cfg(feature = "ffi")]
use jni::JNIEnv;

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeVersion(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    match env.new_string(env!("CARGO_PKG_VERSION")) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeNew(
    mut env: JNIEnv,
    _class: JClass,
    det_model_path: JString,
    rec_model_path: JString,
    dict_path: JString,
) -> jlong {
    let det: String = match env.get_string(&det_model_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    let rec: String = match env.get_string(&rec_model_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    let dict: String = match env.get_string(&dict_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    let config = RustOConfig::new_ppv5(det, rec, dict);
    match RustO::new(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })) as jlong,
        Err(_) => 0,
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOcrFile(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_path: JString,
    results_out: JLongArray,
    count_out: JIntArray,
) -> jint {
    if handle == 0 {
        return -1;
    }

    let path: String = match env.get_string(&image_path) {
        Ok(s) => s.into(),
        Err(_) => return -2,
    };

    let handle_ptr = handle as *mut ROCRHandle;
    let ocr = &mut (*handle_ptr).inner;

    let results = match ocr.run(&path) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(&results);
    let count = c_results.len();
    let results_ptr = c_results.as_ptr() as jlong;
    std::mem::forget(c_results);

    let count_val = [count as i32];
    let results_val = [results_ptr];

    if env.set_long_array_region(&results_out, 0, &results_val).is_err() {
        return -4;
    }
    if env.set_int_array_region(&count_out, 0, &count_val).is_err() {
        return -4;
    }

    0
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOcrData(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_data: JByteArray,
    results_out: JLongArray,
    count_out: JIntArray,
) -> jint {
    if handle == 0 {
        return -1;
    }

    let bytes = match env.convert_byte_array(&image_data) {
        Ok(b) => b,
        Err(_) => return -2,
    };

    #[cfg(not(feature = "use-opencv"))]
    let img = match image::load_from_memory(&bytes) {
        Ok(dynamic_img) => Mat::new(dynamic_img),
        Err(_) => return -3,
    };

    #[cfg(feature = "use-opencv")]
    let img = {
        return -4;
    };

    let handle_ptr = handle as *mut ROCRHandle;
    let ocr = &mut (*handle_ptr).inner;

    let results = match ocr.run_on_mat(&img) {
        Ok(r) => r,
        Err(_) => return -3,
    };

    let c_results = results_to_c(&results);
    let count = c_results.len();
    let results_ptr = c_results.as_ptr() as jlong;
    std::mem::forget(c_results);

    let count_val = [count as i32];
    let results_val = [results_ptr];

    if env.set_long_array_region(&results_out, 0, &results_val).is_err() {
        return -4;
    }
    if env.set_int_array_region(&count_out, 0, &count_val).is_err() {
        return -4;
    }

    0
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeGetResult(
    env: JNIEnv,
    _class: JClass,
    results_ptr: jlong,
    index: jint,
    text_out: JObjectArray,
    score_out: JFloatArray,
    box_out: JFloatArray,
) {
    if results_ptr == 0 || index < 0 {
        return;
    }

    let item_ptr = (results_ptr as *const CTextResult).add(index as usize);
    let item = &*item_ptr;

    if !item.text.is_null() {
        let text_str = CStr::from_ptr(item.text).to_string_lossy();
        if let Ok(j_str) = env.new_string(text_str) {
            let _ = env.set_object_array_element(&text_out, 0, j_str);
        }
    }

    let score_val = [item.score as jfloat];
    let _ = env.set_float_array_region(&score_out, 0, &score_val);

    let box_val = [
        item.box_x1 as jfloat,
        item.box_y1 as jfloat,
        item.box_x2 as jfloat,
        item.box_y2 as jfloat,
        item.box_x3 as jfloat,
        item.box_y3 as jfloat,
        item.box_x4 as jfloat,
        item.box_y4 as jfloat,
    ];
    let _ = env.set_float_array_region(&box_out, 0, &box_val);
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeFreeResults(
    _env: JNIEnv,
    _class: JClass,
    results_ptr: jlong,
    count: jint,
) {
    if results_ptr != 0 && count > 0 {
        rocr_free_results(results_ptr as *mut CTextResult, count as usize);
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeNewWithConfig(
    mut env: JNIEnv,
    _class: JClass,
    config_json: JString,
) -> jlong {
    let json_str: String = match env.get_string(&config_json) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    let config = match RustOConfig::from_json(&json_str) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    match RustO::new(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })) as jlong,
        Err(_) => 0,
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOcrFileWithOutput(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_path: JString,
) -> jlong {
    if handle == 0 {
        return 0;
    }
    let path: String = match env.get_string(&image_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    let handle_ptr = handle as *mut ROCRHandle;
    let ocr = &mut (*handle_ptr).inner;
    match ocr.run(&path) {
        Ok(r) => Box::into_raw(Box::new(ROCROutputHandle { inner: r })) as jlong,
        Err(_) => 0,
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOcrDataWithOutput(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_data: JByteArray,
) -> jlong {
    if handle == 0 {
        return 0;
    }
    let bytes = match env.convert_byte_array(&image_data) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    #[cfg(not(feature = "use-opencv"))]
    let img = match image::load_from_memory(&bytes) {
        Ok(dynamic_img) => Mat::new(dynamic_img),
        Err(_) => return 0,
    };
    #[cfg(feature = "use-opencv")]
    let img = return 0;

    let handle_ptr = handle as *mut ROCRHandle;
    let ocr = &mut (*handle_ptr).inner;
    match ocr.run_on_mat(&img) {
        Ok(r) => Box::into_raw(Box::new(ROCROutputHandle { inner: r })) as jlong,
        Err(_) => 0,
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOutputToRaw(
    env: JNIEnv,
    _class: JClass,
    output: jlong,
) -> jstring {
    if output == 0 {
        return std::ptr::null_mut();
    }
    let output_ref = &(*(output as *const ROCROutputHandle)).inner;
    match env.new_string(output_ref.to_raw()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOutputToCsv(
    env: JNIEnv,
    _class: JClass,
    output: jlong,
) -> jstring {
    if output == 0 {
        return std::ptr::null_mut();
    }
    let output_ref = &(*(output as *const ROCROutputHandle)).inner;
    match env.new_string(output_ref.to_csv()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOutputToTextWithPosition(
    env: JNIEnv,
    _class: JClass,
    output: jlong,
) -> jstring {
    if output == 0 {
        return std::ptr::null_mut();
    }
    let output_ref = &(*(output as *const ROCROutputHandle)).inner;
    match env.new_string(output_ref.to_text_with_position()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeOutputToSpatialText(
    env: JNIEnv,
    _class: JClass,
    output: jlong,
    y_threshold_multiplier: jfloat,
    x_threshold_multiplier: jfloat,
) -> jstring {
    if output == 0 {
        return std::ptr::null_mut();
    }
    let output_ref = &(*(output as *const ROCROutputHandle)).inner;
    let y_mult = if y_threshold_multiplier <= 0.0 {
        None
    } else {
        Some(y_threshold_multiplier as f32)
    };
    let x_mult = if x_threshold_multiplier <= 0.0 {
        None
    } else {
        Some(x_threshold_multiplier as f32)
    };
    match env.new_string(output_ref.to_spatial_text(y_mult, x_mult)) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeFreeOutput(
    _env: JNIEnv,
    _class: JClass,
    output: jlong,
) {
    if output != 0 {
        rocr_free_output(output as *mut ROCROutputHandle);
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        rocr_free(handle as *mut ROCRHandle);
    }
}

