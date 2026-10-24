"""RustO - High performance OCR engine powered by Rust."""

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
    "OcrRunOptions",
    "DetectionRunOptions",
    "PostprocessRunOptions",
    "Frame",
    "RustoBindingError",
]
