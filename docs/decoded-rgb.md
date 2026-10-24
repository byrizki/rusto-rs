# OCR from decoded RGB pixels

Callers such as libvips image servers already have decoded pixels. Encoding these
pixels as JPEG or PNG solely to call `rocr_detect_text_data` adds codec work and,
for JPEG, another lossy transformation. `rocr_detect_text_rgb` bypasses both
codecs while using the same detector, recognizer, options, and coordinates.
It requires the default pure-Rust image backend; the optional OpenCV backend
returns an engine/backend error, consistent with encoded in-memory input.

Build with `cargo build --release --features ffi`. C declarations are in
`include/rusto.h`. Existing encoded-image functions and their ABI are unchanged.

```c
#include "rusto.h"

/* handle has already been initialized; pixels is caller-owned RGB8. */
CTextResult *results = NULL;
size_t count = 0;
int status = rocr_detect_text_rgb(handle, pixels, buffer_length,
    width, height, stride_bytes, "{}", &results, &count);
if (status == 0) {
    /* Consume/copy results here. */
    rocr_free_results(results, count);
}
```

The layout is interleaved unsigned 8-bit **RGB**, not BGR, RGBA, planar, or
premultiplied pixels. Convert color space, remove alpha, and apply orientation
before calling when needed. Rows can contain padding; `stride_bytes` must be at
least `width * 3`. The buffer must contain at least
`(height - 1) * stride_bytes + width * 3` readable bytes. Zero dimensions,
unsupported dimensions, integer overflow, insufficient stride, and short buffers
are rejected before a Rust slice is constructed. As with other C APIs, the caller
must supply truthful allocation bounds and valid, nonaliasing pointers.

RustO copies the active row bytes into owned image storage. This is **not
zero-copy**; it eliminates encoding and decoding. The input pointer is not
retained, modified, or freed. One handle must have only one call in flight;
use separate handles for concurrent workers. Results are in the supplied image
coordinate frame. The input has no metadata, so EXIF is not interpreted.

Return codes: `0` success; `-1` null pointer; `-2` invalid layout/length;
`-3` engine/backend failure; `-4` invalid options; `-5` spatial output requested.
Non-null output slots are reset to null/zero on error. Structured `lines` and
`words` outputs use `rocr_free_results`; a C spatial RGB API is not introduced.
The Rust method `RustO::detect_text_rgb(&[u8], width, height, stride, &options)`
supports all existing output granularities, including spatial.

## Validation

Layout and null-pointer tests need no models:

```sh
cargo test --features ffi rgb_layout --lib
cargo test --features ffi --test rgb_input_test
```

The opt-in model-backed test checks known text, parity with lossless PNG input,
padded rows with no final-row padding, unchanged source bytes, short buffers,
invalid options/output modes, repeated calls, and freeing C results:

```sh
RUSTO_TEST_MODELS=/absolute/path/to/ppocrv6-models \
  cargo test --features ffi --test rgb_input_test -- --include-ignored
```

The model directory must contain `det.onnx`, `rec.onnx`, and `dict.txt`.
The fixture is generated text, not a third-party document.
