//
//  RustO.h
//  RustO iOS / macOS SDK
//
//  Objective-C / C interface for RustO OCR engine.
//

#ifndef RUSTO_H
#define RUSTO_H

#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

/// C-compatible text result representation from RustO OCR engine.
typedef struct {
    char *text;
    float score;
    float box_x1;
    float box_y1;
    float box_x2;
    float box_y2;
    float box_x3;
    float box_y3;
    float box_x4;
    float box_y4;
    float frame_width;
    float frame_height;
    float frame_top;
    float frame_left;
} ROCRTextResult;

/// Opaque pointer handle to a RustO OCR engine instance.
typedef void *ROCRHandle;

/// Initializes a new RustO OCR engine instance with JSON configuration.
/// Returns NULL on initialization failure.
ROCRHandle rocr_initialize(const char *config_json);

/// Detects text from an image file at the specified path.
/// Returns 0 on success, negative error code on failure.
int rocr_detect_text_file(ROCRHandle handle, const char *image_path, const char *options_json, void **results, size_t *count);

/// Detects text from an image file and returns layout-preserved spatial text.
/// Caller must free returned string using rocr_free_string.
char *rocr_detect_text_file_spatial(ROCRHandle handle, const char *image_path, const char *options_json);

/// Detects text from in-memory image bytes.
/// Returns 0 on success, negative error code on failure.
int rocr_detect_text_data(ROCRHandle handle, const void *data, size_t length, const char *options_json, void **results, size_t *count);

/// Detects text from in-memory image bytes and returns layout-preserved spatial text.
/// Caller must free returned string using rocr_free_string.
char *rocr_detect_text_data_spatial(ROCRHandle handle, const void *data, size_t length, const char *options_json);

/// Frees a string returned by spatial detection functions.
void rocr_free_string(char *s);

/// Frees results array allocated by rocr_detect_text_file or rocr_detect_text_data.
void rocr_free_results(void *results, size_t count);

/// Frees a RustO engine handle and associated resources.
void rocr_free(ROCRHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* RUSTO_H */
