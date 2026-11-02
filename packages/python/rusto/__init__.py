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
    InitializeConfig as _GeneratedInitializeConfig,
    DetectionConfig as _GeneratedDetectionConfig,
    RecognitionConfig as _GeneratedRecognitionConfig,
    ClassificationConfig as _GeneratedClassificationConfig,
    OrientationConfig as _GeneratedOrientationConfig,
    UnwarpConfig as _GeneratedUnwarpConfig,
    LayoutConfig as _GeneratedLayoutConfig,
    LayoutType,
    LayoutBBox,
    LayoutRegion,
    UnwarpedImageResult,
    CalibrationOptions,
    OcrRunOptions as _GeneratedOcrRunOptions,
    DetectionRunOptions,
    PostprocessRunOptions,
    Frame,
    RustoBindingError,
    ImageSource,
    OutputGranularity,
    TextResult,
    DetectTextResult,
    OptimizationOptions as _GeneratedOptimizationOptions,
    default_ocr_run_options,
    create_initialize_config_ppv6,
    create_initialize_config_for_model as _generated_create_initialize_config_for_model,
    download_model as _generated_download_model,
    download_all_models as _generated_download_all_models,
    rusto_version,
)


class DetectionConfig(_GeneratedDetectionConfig):
    """Text detection model configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        use_dilation: bool | None = None,
        score_mode: str | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            use_dilation=use_dilation,
            score_mode=score_mode,
        )


class RecognitionConfig(_GeneratedRecognitionConfig):
    """Text recognition model configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        dict_path: str | None = None,
        rec_batch_num: int | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            dict_path=dict_path,
            rec_batch_num=rec_batch_num,
        )


class ClassificationConfig(_GeneratedClassificationConfig):
    """Text line orientation classification configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        thresh: float | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            thresh=thresh,
        )


class OrientationConfig(_GeneratedOrientationConfig):
    """Document orientation classification configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        thresh: float | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            thresh=thresh,
        )


class UnwarpConfig(_GeneratedUnwarpConfig):
    """Document dewarping configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
        )


class LayoutConfig(_GeneratedLayoutConfig):
    """Document layout analysis configuration."""
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        conf_thresh: float | None = None,
        target_size: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            conf_thresh=conf_thresh,
            target_size=target_size,
            target_height=target_height,
            target_width=target_width,
        )


class InitializeConfig(_GeneratedInitializeConfig):
    """Engine initialization configuration specifying model paths, presets, and options."""
    def __init__(
        self,
        model: str | None = None,
        model_path: str | None = None,
        template: str | None = None,
        lang: str | None = None,
        cache_dir: str | None = None,
        enable_reading_order: bool | None = None,
        layout_variant: str | None = None,
        detection: DetectionConfig | None = None,
        recognition: RecognitionConfig | None = None,
        classification: ClassificationConfig | None = None,
        orientation: OrientationConfig | None = None,
        unwarp: UnwarpConfig | None = None,
        layout: LayoutConfig | None = None,
    ):
        super().__init__(
            model=model,
            model_path=model_path,
            template=template,
            lang=lang,
            cache_dir=cache_dir,
            enable_reading_order=enable_reading_order,
            layout_variant=layout_variant,
            detection=detection,
            recognition=recognition,
            classification=classification,
            orientation=orientation,
            unwarp=unwarp,
            layout=layout,
        )


class OptimizationOptions(_GeneratedOptimizationOptions):
    """Runtime crop and resolution optimization options."""
    def __init__(
        self,
        enabled: bool | None = None,
        target_max_side: int | None = None,
        crop_padding_x: float | None = None,
        crop_padding_y: float | None = None,
    ):
        super().__init__(
            enabled=enabled,
            target_max_side=target_max_side,
            crop_padding_x=crop_padding_x,
            crop_padding_y=crop_padding_y,
        )


class OcrRunOptions(_GeneratedOcrRunOptions):
    """Per-request execution options for text detection."""
    def __init__(
        self,
        output: OutputGranularity = OutputGranularity.LINES,
        line_y_threshold: float | None = None,
        word_x_threshold: float | None = None,
        text_score: float | None = None,
        classification: bool | None = None,
        orientation: bool | None = None,
        unwarp: bool | None = None,
        layout: bool | None = None,
        min_height: float | None = None,
        max_side_len: float | None = None,
        min_side_len: float | None = None,
        width_height_ratio: float | None = None,
        detection: DetectionRunOptions | None = None,
        postprocess: PostprocessRunOptions | None = None,
        calibration: CalibrationOptions | None = None,
        optimization: OptimizationOptions | None = None,
        rec_batch_num: int | None = None,
    ):
        super().__init__(
            output=output,
            line_y_threshold=line_y_threshold,
            word_x_threshold=word_x_threshold,
            text_score=text_score,
            classification=classification,
            orientation=orientation,
            unwarp=unwarp,
            layout=layout,
            min_height=min_height,
            max_side_len=max_side_len,
            min_side_len=min_side_len,
            width_height_ratio=width_height_ratio,
            detection=detection,
            postprocess=postprocess,
            calibration=calibration,
            optimization=optimization,
            rec_batch_num=rec_batch_num,
        )


def create_initialize_config_for_model(
    model: str | None = None,
    lang: str | None = None,
    model_path: str | None = None,
) -> InitializeConfig:
    """Creates an InitializeConfig for a canonical model identifier and optional model directory."""
    raw = _generated_create_initialize_config_for_model(model, lang, model_path)
    return InitializeConfig(
        model=raw.model,
        model_path=raw.model_path,
        template=raw.template,
        lang=raw.lang,
        cache_dir=raw.cache_dir,
        enable_reading_order=raw.enable_reading_order,
        layout_variant=raw.layout_variant,
        detection=raw.detection,
        recognition=raw.recognition,
        classification=raw.classification,
        orientation=raw.orientation,
        unwarp=raw.unwarp,
        layout=raw.layout,
    )


def download_model(
    model: str | None = None,
    lang: str | None = None,
    cache_dir: str | None = None,
) -> list[str]:
    """Downloads model assets for a specific OCR pipeline ahead of time."""
    return _generated_download_model(model, lang, cache_dir)


def download_all_models(cache_dir: str | None = None) -> list[str]:
    """Downloads all available canonical models to the cache directory ahead of time."""
    return _generated_download_all_models(cache_dir)


RustO.download_model = staticmethod(download_model)
RustO.download_all_models = staticmethod(download_all_models)


__all__ = [
    "RustO",
    "RustOProtocol",
    "InitializeConfig",
    "DetectionConfig",
    "RecognitionConfig",
    "ClassificationConfig",
    "OrientationConfig",
    "UnwarpConfig",
    "LayoutConfig",
    "LayoutType",
    "LayoutBBox",
    "LayoutRegion",
    "UnwarpedImageResult",
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
    "default_ocr_run_options",
    "create_initialize_config_ppv6",
    "create_initialize_config_for_model",
    "download_model",
    "download_all_models",
    "rusto_version",
]
