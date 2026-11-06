# rusto (Python SDK)

Official Python bindings for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the RTen inference runtime via UniFFI.

## Features

- **Zero-Config Indra OCR**: `RustO.initialize()` automatically downloads and caches the default `indraocr` `.rten` models on first run.
- **Context Manager Support**: Deterministic native resource cleanup with `with RustO.initialize() as engine:` or `engine.close()`.
- **Full Document Pipeline**: Neural document unwarping (`with_auto_unwarp()`), page orientation detection (`with_orientation_detection()`), semantic layout analysis, and reading-order recovery (`with_layout_reading()`).
- **Fluent Builder API**: Chain `.as_lines()`, `.as_words()`, `.as_spatial()`, or `.with_output(...)` into a single `.run()` terminal method.
- **Zero-Copy Raw RGB Support**: Feed NumPy / OpenCV / Pillow RGB8 buffers directly via `.rgb(pixels, width, height, stride)`.

## Installation

```bash
pip install rusto
```

## Quick Start

```python
from rusto import RustO

# Zero-config initialization (auto-downloads Indra OCR .rten models if not cached)
with RustO.initialize() as engine:
    result = (
        engine.detect_text("invoice.png")
        .with_auto_unwarp()
        .with_layout_reading()
        .as_lines()
        .run()
    )

    if result.is_STRUCTURED():
        for item in result.results:
            print(
                f"[{item.score:.2f}] ({item.layout_type or 'line'} #{item.region_id}) {item.text}"
            )

    # Reconstructed 2D spatial text layout
    spatial = engine.detect_text("invoice.png").as_spatial().run()
    if spatial.is_SPATIAL():
        print(spatial.text)
```

You can also manage the engine lifecycle explicitly without a `with` block:

```python
from rusto import InitializeConfig, RustO

engine = RustO.initialize(InitializeConfig.indra())
try:
    result = engine.detect_text("invoice.png").as_words().run()
    if result.is_STRUCTURED():
        for word in result.results:
            print(word.text, word.frame)
finally:
    engine.close()
```

## Custom Configuration & Offline `.rten` Models

Select canonical model tiers (`"indraocr"`, `"ppocrv6-tiny"`, `"ppocrv6-small"`, `"ppocrv5-mobile"`, `"ppocrv4-mobile"`), languages, or local `.rten` model paths:

```python
from rusto import (
    DetectionConfig,
    InitializeConfig,
    LayoutConfig,
    RecognitionConfig,
    RustO,
    UnwarpConfig,
)

# Preset by canonical model + language (auto-downloaded to cache)
cfg = InitializeConfig.for_model("ppocrv6-tiny", lang="latin")
engine = RustO.initialize(cfg)
engine.close()

# Explicit local .rten assets for offline environments
offline_cfg = InitializeConfig(
    template="indra",
    enable_reading_order=True,
    detection=DetectionConfig(enabled=True, model_path="./models/indra_det.rten"),
    recognition=RecognitionConfig(
        enabled=True,
        model_path="./models/indra_rec.rten",
        dict_path="./models/indra_meta.json",
    ),
    unwarp=UnwarpConfig(enabled=True, model_path="./models/uvdoc.rten"),
    layout=LayoutConfig(enabled=True, model_path="./models/layout_cdla.rten"),
)
with RustO.initialize(offline_cfg) as offline_engine:
    res = offline_engine.detect_text("scan.png").with_auto_unwarp().as_lines().run()
```

Pre-download models into a target cache directory ahead of time (e.g., during Docker image build):

```python
from rusto import download_model

paths = download_model(model="indraocr", lang="latin", cache_dir="./models-cache")
```

## Input Sources & Raw RGB Buffers

Pass a file path (`str` or `Path`), encoded image `bytes`, `ImageSource`, or raw interleaved 24-bit RGB pixels:

```python
from pathlib import Path
import numpy as np
from rusto import RustO

with RustO.initialize() as engine:
    # 1. Encoded PNG/JPEG bytes
    png_bytes = Path("invoice.png").read_bytes()
    res_bytes = engine.detect_text(png_bytes).as_lines().run()

    # 2. Raw HxWx3 uint8 RGB array (e.g. from Pillow / imageio)
    rgb_array = np.zeros((720, 1280, 3), dtype=np.uint8)
    height, width, _ = rgb_array.shape
    stride = width * 3

    res_rgb = (
        engine.detect_text()
        .rgb(rgb_array.tobytes(), width, height, stride)
        .as_lines()
        .run()
    )
```

## 6-Phase Pipeline Options

Customize every phase of OCR execution per request:

```python
from rusto import (
    DetectionRunOptions,
    OutputGranularity,
    PostDetectionOptions,
    PostRecognitionOptions,
    PreDetectionOptions,
    PreRecognitionOptions,
    RecognitionRunOptions,
)

result = (
    engine.detect_text("document.jpg")
    .with_orientation_detection()
    .with_auto_unwarp()
    .with_layout_reading()
    .with_pre_detection(PreDetectionOptions(descreen_strength=0.3, target_max_side=1280))
    .with_detection(DetectionRunOptions(limit_side_len=960, limit_type="max"))
    .with_post_detection(PostDetectionOptions(threshold=0.3, box_threshold=0.6, unclip_ratio=1.6))
    .with_pre_recognition(PreRecognitionOptions(line_y_threshold=0.5, word_x_threshold=0.4))
    .with_recognition(RecognitionRunOptions(batch_size=8))
    .with_post_recognition(PostRecognitionOptions(score_threshold=0.5))
    .with_output(OutputGranularity.LINES)
    .run()
)
```

## Result Reference

`DetectTextResult` is a tagged union returned by `.run()`:

- `result.is_STRUCTURED()` -> `True` for `.as_lines()` and `.as_words()`. Access `result.results` (`list[TextResult]`):
  - `item.text` (`str`): Recognized text string.
  - `item.score` (`float`): Confidence score (`0.0` – `1.0`).
  - `item.box_points` (`list[Point]`): 4 polygon vertices (`x`, `y`).
  - `item.frame` (`Frame`): Axis-aligned bounding box (`width`, `height`, `top`, `left`).
  - `item.region_id` (`int | None`): Layout region index when `with_layout_reading()` is active.
  - `item.layout_type` (`LayoutType | None`): Semantic region category (`TEXT`, `TITLE`, `TABLE`, `HEADER`, `FOOTER`, `FIGURE`, etc.).
- `result.is_SPATIAL()` -> `True` for `.as_spatial()`. Access `result.text` (`str`).

## License

Apache-2.0 OR MIT
