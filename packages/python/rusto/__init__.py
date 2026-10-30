"""RustO - High performance, pure-Rust OCR engine for Python.

RustO provides fast, memory-safe text extraction based on PaddleOCR architectures
and the pure-Rust RTen deep learning inference engine.

Quick Start:
    >>> from rusto import RustO, InitializeConfig, OcrRunOptions, ImageSource
    >>> config = InitializeConfig.ppv6("models/det.rten", "models/rec.rten", "models/dict.txt")
    >>> engine = RustO.initialize(config)
    >>> result = engine.detect_text(ImageSource.PATH("receipt.jpg"), OcrRunOptions())
    >>> if result.is_STRUCTURED():
    ...     for item in result.results:
    ...         print(f"[{item.score:.2f}] {item.text}")
    >>> engine.close()

Exported Classes & Types:
    - RustO: Main OCR engine class with initialize(), detect_text(), detect_text_rgb(), and close().
    - InitializeConfig: Initial session configuration (presets, model paths, execution threads).
    - OcrRunOptions: Per-request execution options (output shape, thresholds, resize limits).
    - TextResult: Recognized text string with confidence score, quad vertices, and bounding frame.
    - Frame: Axis-aligned bounding box frame (width, height, top, left).
    - ImageSource: Input image specification (PATH, BYTES).
    - OutputGranularity: LINES, WORDS, or SPATIAL.
    - DetectTextResult: STRUCTURED (results: list[TextResult]) or SPATIAL (text: str).
"""

from __future__ import annotations
import os
import sys

# Ensure host library directory is in search path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_target_debug = os.path.abspath(os.path.join(_current_dir, "../../../target/debug"))
_repo_target_release = os.path.abspath(os.path.join(_current_dir, "../../../target/release"))

for _candidate in [_current_dir, _repo_target_debug, _repo_target_release]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# If library is not present in generated directory, link from target build directories if available
_generated_dir = os.path.join(_current_dir, "generated")
_libname = (
    "librusto.dylib"
    if sys.platform == "darwin"
    else ("rusto.dll" if sys.platform in ("win32", "cygwin") else "librusto.so")
)
_target_lib = os.path.join(_generated_dir, _libname)

if not os.path.isfile(_target_lib):
    for _candidate_dir in [_repo_target_release, _repo_target_debug]:
        _candidate_file = os.path.join(_candidate_dir, _libname)
        if os.path.isfile(_candidate_file):
            try:
                os.symlink(_candidate_file, _target_lib)
                break
            except OSError:
                try:
                    import shutil
                    shutil.copy2(_candidate_file, _target_lib)
                    break
                except OSError:
                    pass

from .generated.rusto import (
    RustO,
    RustOProtocol,
    InitializeConfig,
    DetectionConfig,
    RecognitionConfig,
    ClassificationConfig,
    OrientationConfig,
    UnwarpConfig,
    CalibrationOptions,
    OcrRunOptions,
    DetectionRunOptions,
    PostprocessRunOptions,
    Frame,
    RustoBindingError,
    ImageSource,
    OutputGranularity,
    TextResult,
    DetectTextResult,
    OptimizationOptions,
)

__all__ = [
    "RustO",
    "RustOProtocol",
    "InitializeConfig",
    "DetectionConfig",
    "RecognitionConfig",
    "ClassificationConfig",
    "OrientationConfig",
    "UnwarpConfig",
    "CalibrationOptions",
    "OptimizationOptions",
    "OcrRunOptions",
    "DetectionRunOptions",
    "PostprocessRunOptions",
    "Frame",
    "RustoBindingError",
    "ImageSource",
    "OutputGranularity",
    "TextResult",
    "DetectTextResult",
]
