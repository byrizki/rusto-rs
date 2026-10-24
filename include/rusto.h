#ifndef RUSTO_H
#define RUSTO_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ROCRHandle ROCRHandle;
typedef struct CTextResult {
    char *text;
    float score;
    float box_x1, box_y1, box_x2, box_y2;
    float box_x3, box_y3, box_x4, box_y4;
    float frame_width, frame_height, frame_top, frame_left;
} CTextResult;

ROCRHandle *rocr_initialize(const char *config_json);
void rocr_free(ROCRHandle *handle);
const char *rocr_version(void);
int rocr_detect_text_file(ROCRHandle *, const char *, const char *, CTextResult **, size_t *);
int rocr_detect_text_data(ROCRHandle *, const uint8_t *, size_t, const char *, CTextResult **, size_t *);
char *rocr_detect_text_file_spatial(ROCRHandle *, const char *, const char *);
char *rocr_detect_text_data_spatial(ROCRHandle *, const uint8_t *, size_t, const char *);
void rocr_free_results(CTextResult *results, size_t count);
void rocr_free_string(char *text);

/* Decoded interleaved RGB8, no alpha. stride is bytes between row starts.
 * image_len must cover (height - 1) * stride + width * 3 bytes in one readable
 * allocation. The final row need not include padding. No EXIF rotation occurs.
 * The caller owns pixels; RustO copies them and never retains the pointer.
 * The handle must not be used concurrently. options_json is required ("{}").
 * results/count are reset on failure and must be valid nonaliasing output slots.
 * Return: 0 success; -1 null pointer; -2 invalid layout/length; -3 engine/backend
 * error; -4 invalid options; -5 spatial output requested (only lines/words).
 * Pure Rust image backend only. Free results with rocr_free_results.
 */
int rocr_detect_text_rgb(ROCRHandle *handle, const uint8_t *pixels, size_t image_len,
                         uint32_t width, uint32_t height, size_t stride,
                         const char *options_json, CTextResult **results, size_t *count);
#ifdef __cplusplus
}
#endif
#endif
