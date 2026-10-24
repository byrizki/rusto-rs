#ifndef RUSTO_GO_H
#define RUSTO_GO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
} CTextResult;

typedef void *ROCRHandle;

ROCRHandle rocr_initialize(const char *config_json);
int rocr_detect_text_file(ROCRHandle handle, const char *image_path, const char *options_json, void **results, size_t *count);
char *rocr_detect_text_file_spatial(ROCRHandle handle, const char *image_path, const char *options_json);
int rocr_detect_text_data(ROCRHandle handle, const void *data, size_t length, const char *options_json, void **results, size_t *count);
char *rocr_detect_text_data_spatial(ROCRHandle handle, const void *data, size_t length, const char *options_json);
int rocr_detect_text_rgb(ROCRHandle handle, const void *pixels, size_t image_len, uint32_t width, uint32_t height, size_t stride, const char *options_json, void **results, size_t *count);
char *rocr_detect_text_rgb_spatial(ROCRHandle handle, const void *pixels, size_t image_len, uint32_t width, uint32_t height, size_t stride, const char *options_json);
void rocr_free_string(char *s);
void rocr_free_results(void *results, size_t count);
void rocr_free(ROCRHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* RUSTO_GO_H */
