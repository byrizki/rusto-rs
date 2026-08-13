// FFI bindings for C/C++/C#
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
    if options
        .line_y_threshold
        .is_some_and(|value| !value.is_finite() || value < 0.0)
        || options
            .word_x_threshold
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || options
            .text_score
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(ROCR_INVALID_OPTIONS);
    }
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
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);
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
    let options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    if options.output != OutputGranularity::Spatial {
        return std::ptr::null_mut();
    }
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
    *results_out = c_results.as_ptr() as *mut CTextResult;
    std::mem::forget(c_results);
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
    let options = match parse_run_options(options_json) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    if options.output != OutputGranularity::Spatial {
        return std::ptr::null_mut();
    }
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

fn text_results_to_c(results: &[crate::TextResult]) -> Vec<CTextResult> {
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
        .collect()
}

// ============================================================================
// Android JNI bindings (com.byrizki.rusto.RustO)
// ============================================================================

#[cfg(feature = "ffi")]
use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray, JLongArray, JObjectArray, JString};
#[cfg(feature = "ffi")]
use jni::sys::{jfloat, jint, jlong, jstring};
#[cfg(feature = "ffi")]
use jni::JNIEnv;

#[cfg(feature = "ffi")]
unsafe fn write_jni_results(
    env: &mut JNIEnv,
    results_out: &JLongArray,
    count_out: &JIntArray,
    results: Vec<CTextResult>,
) -> jint {
    let count = results.len();
    let ptr = results.as_ptr() as jlong;
    if env.set_long_array_region(results_out, 0, &[ptr]).is_err()
        || env.set_int_array_region(count_out, 0, &[count as jint]).is_err()
    {
        // `results` still owns allocations here; normal Vec drop releases the
        // buffer but not individual CString allocations.
        for result in results {
            if !result.text.is_null() {
                drop(CString::from_raw(result.text));
            }
        }
        return -4;
    }
    std::mem::forget(results);
    0
}

#[cfg(feature = "ffi")]
unsafe fn parse_jni_options(env: &mut JNIEnv, options_json: JString) -> Result<OcrRunOptions, jint> {
    let json: String = env.get_string(&options_json).map_err(|_| -4)?.into();
    let c_json = CString::new(json).map_err(|_| ROCR_INVALID_OPTIONS)?;
    parse_run_options(c_json.as_ptr()).map_err(|code| code as jint)
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeInitialize(
    mut env: JNIEnv,
    _class: JClass,
    config_json: JString,
) -> jlong {
    let json: String = match env.get_string(&config_json) { Ok(value) => value.into(), Err(_) => return 0 };
    let config = match InitializeConfig::from_json(&json) { Ok(value) => value, Err(_) => return 0 };
    match RustO::initialize(config) {
        Ok(ocr) => Box::into_raw(Box::new(ROCRHandle { inner: ocr })) as jlong,
        Err(_) => 0,
    }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeDetectTextFile(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_path: JString,
    options_json: JString,
    results_out: JLongArray,
    count_out: JIntArray,
) -> jint {
    if handle == 0 { return -1; }
    let path: String = match env.get_string(&image_path) { Ok(value) => value.into(), Err(_) => return -2 };
    let options = match parse_jni_options(&mut env, options_json) { Ok(value) => value, Err(code) => return code };
    if options.output == OutputGranularity::Spatial { return -5; }
    let detected = match (&mut (*(handle as *mut ROCRHandle)).inner)
        .detect_text(&ImageSource::Path(path.into()), &options)
    { Ok(value) => value, Err(_) => return -3 };
    let DetectTextResult::Structured(results) = detected else { return -5; };
    write_jni_results(&mut env, &results_out, &count_out, text_results_to_c(&results))
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeDetectTextData(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_data: JByteArray,
    options_json: JString,
    results_out: JLongArray,
    count_out: JIntArray,
) -> jint {
    if handle == 0 { return -1; }
    let bytes = match env.convert_byte_array(&image_data) { Ok(value) => value, Err(_) => return -2 };
    let options = match parse_jni_options(&mut env, options_json) { Ok(value) => value, Err(code) => return code };
    if options.output == OutputGranularity::Spatial { return -5; }
    let detected = match (&mut (*(handle as *mut ROCRHandle)).inner)
        .detect_text(&ImageSource::Bytes(bytes), &options)
    { Ok(value) => value, Err(_) => return -3 };
    let DetectTextResult::Structured(results) = detected else { return -5; };
    write_jni_results(&mut env, &results_out, &count_out, text_results_to_c(&results))
}

#[cfg(feature = "ffi")]
unsafe fn spatial_to_jstring(env: &mut JNIEnv, detected: DetectTextResult) -> jstring {
    let DetectTextResult::Spatial(text) = detected else { return std::ptr::null_mut(); };
    env.new_string(text).map_or(std::ptr::null_mut(), |value| value.into_raw())
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeDetectTextFileSpatial(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_path: JString,
    options_json: JString,
) -> jstring {
    if handle == 0 { return std::ptr::null_mut(); }
    let path: String = match env.get_string(&image_path) { Ok(value) => value.into(), Err(_) => return std::ptr::null_mut() };
    let options = match parse_jni_options(&mut env, options_json) { Ok(value) => value, Err(_) => return std::ptr::null_mut() };
    if options.output != OutputGranularity::Spatial { return std::ptr::null_mut(); }
    let detected = match (&mut (*(handle as *mut ROCRHandle)).inner)
        .detect_text(&ImageSource::Path(path.into()), &options)
    { Ok(value) => value, Err(_) => return std::ptr::null_mut() };
    spatial_to_jstring(&mut env, detected)
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeDetectTextDataSpatial(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_data: JByteArray,
    options_json: JString,
) -> jstring {
    if handle == 0 { return std::ptr::null_mut(); }
    let bytes = match env.convert_byte_array(&image_data) { Ok(value) => value, Err(_) => return std::ptr::null_mut() };
    let options = match parse_jni_options(&mut env, options_json) { Ok(value) => value, Err(_) => return std::ptr::null_mut() };
    if options.output != OutputGranularity::Spatial { return std::ptr::null_mut(); }
    let detected = match (&mut (*(handle as *mut ROCRHandle)).inner)
        .detect_text(&ImageSource::Bytes(bytes), &options)
    { Ok(value) => value, Err(_) => return std::ptr::null_mut() };
    spatial_to_jstring(&mut env, detected)
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
    if results_ptr == 0 || index < 0 { return; }
    let item = &*((results_ptr as *const CTextResult).add(index as usize));
    if !item.text.is_null() {
        if let Ok(text) = env.new_string(CStr::from_ptr(item.text).to_string_lossy()) {
            let _ = env.set_object_array_element(&text_out, 0, text);
        }
    }
    let _ = env.set_float_array_region(&score_out, 0, &[item.score as jfloat]);
    let _ = env.set_float_array_region(&box_out, 0, &[
        item.box_x1 as jfloat, item.box_y1 as jfloat, item.box_x2 as jfloat, item.box_y2 as jfloat,
        item.box_x3 as jfloat, item.box_y3 as jfloat, item.box_x4 as jfloat, item.box_y4 as jfloat,
    ]);
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeFreeResults(
    _env: JNIEnv, _class: JClass, results_ptr: jlong, count: jint,
) {
    if results_ptr != 0 && count >= 0 { rocr_free_results(results_ptr as *mut CTextResult, count as usize); }
}

#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn Java_com_byrizki_rusto_RustO_nativeFree(
    _env: JNIEnv, _class: JClass, handle: jlong,
) {
    if handle != 0 { rocr_free(handle as *mut ROCRHandle); }
}
