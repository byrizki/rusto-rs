"""RustO - High performance, pure-Rust OCR engine for Python.

RustO provides fast, memory-safe text extraction powered by PaddleOCR and Indra OCR
architectures on pure-Rust MNN (`rusto-nn`) and RTen (`rusto-rten`) inference engines.

Quick Start:
    >>> from rusto import RustO, InitializeConfig
    >>> engine = RustO.initialize()
    >>> result = engine.detect_text("receipt.jpg").with_auto_unwarp().with_layout_reading().as_lines().run()
    >>> if result.is_STRUCTURED():
    ...     for item in result.results:
    ...         print(f"[{item.score:.2f}] {item.text}")
    >>> engine.close()
"""

from __future__ import annotations
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_target_debug = os.path.abspath(os.path.join(_current_dir, "../../../target/debug"))
_repo_target_release = os.path.abspath(os.path.join(_current_dir, "../../../target/release"))

for _candidate in [_current_dir, _repo_target_debug, _repo_target_release]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

_generated_dir = os.path.join(_current_dir, "generated")
_libname = (
    "librusto.dylib"
    if sys.platform == "darwin"
    else ("rusto.dll" if sys.platform in ("win32", "cygwin") else "librusto.so")
)
_target_lib = os.path.join(_generated_dir, _libname)

for _candidate_dir in [_repo_target_release, _repo_target_debug]:
    _candidate_file = os.path.join(_candidate_dir, _libname)
    if os.path.isfile(_candidate_file):
        try:
            if os.path.islink(_target_lib) or (
                os.path.isfile(_target_lib)
                and os.path.getmtime(_candidate_file) > os.path.getmtime(_target_lib)
            ):
                os.unlink(_target_lib)
            if not os.path.exists(_target_lib):
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
    OcrRunOptions as _GeneratedOcrRunOptions,
    PreDetectionOptions as _GeneratedPreDetectionOptions,
    DetectionRunOptions as _GeneratedDetectionRunOptions,
    PostDetectionOptions as _GeneratedPostDetectionOptions,
    PreRecognitionOptions as _GeneratedPreRecognitionOptions,
    RecognitionRunOptions as _GeneratedRecognitionRunOptions,
    PostRecognitionOptions as _GeneratedPostRecognitionOptions,
    Frame,
    RustoBindingError,
    ImageSource,
    OutputGranularity,
    TextResult,
    DetectTextResult,
    default_ocr_run_options,
    create_initialize_config_indra,
    create_initialize_config_ppv6,
    create_initialize_config_for_model as _generated_create_initialize_config_for_model,
    download_model as _generated_download_model,
    rusto_version,
    _UniffiLib,
    _uniffi_rust_call,
)


class PreDetectionOptions(_GeneratedPreDetectionOptions):
    """Phase 1: Pre-detection options."""
    def __init__(
        self,
        *,
        descreen_strength: float | None = None,
        target_max_side: int | None = None,
        crop_padding_x: float | None = None,
        crop_padding_y: float | None = None,
    ):
        super().__init__(
            descreen_strength=descreen_strength,
            target_max_side=target_max_side,
            crop_padding_x=crop_padding_x,
            crop_padding_y=crop_padding_y,
        )


class DetectionRunOptions(_GeneratedDetectionRunOptions):
    """Phase 2: Detection inference options."""
    def __init__(
        self,
        *,
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        super().__init__(
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            mean=mean,
            std=std,
        )


class PostDetectionOptions(_GeneratedPostDetectionOptions):
    """Phase 3: Post-detection options."""
    def __init__(
        self,
        *,
        threshold: float | None = None,
        box_threshold: float | None = None,
        unclip_ratio: float | None = None,
        use_dilation: bool | None = None,
        score_mode: str | None = None,
        max_candidates: int | None = None,
        min_height: float | None = None,
        min_side_len: float | None = None,
        max_side_len: float | None = None,
        width_height_ratio: float | None = None,
    ):
        super().__init__(
            threshold=threshold,
            box_threshold=box_threshold,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            max_candidates=max_candidates,
            min_height=min_height,
            min_side_len=min_side_len,
            max_side_len=max_side_len,
            width_height_ratio=width_height_ratio,
        )


class PreRecognitionOptions(_GeneratedPreRecognitionOptions):
    """Phase 4: Pre-recognition clustering options."""
    def __init__(
        self,
        *,
        line_y_threshold: float | None = None,
        word_x_threshold: float | None = None,
    ):
        super().__init__(
            line_y_threshold=line_y_threshold,
            word_x_threshold=word_x_threshold,
        )


class RecognitionRunOptions(_GeneratedRecognitionRunOptions):
    """Phase 5: Recognition inference options.

    Args:
        batch_size: Recognition batch size (number of text crops processed concurrently).
        workers: Number of parallel worker threads for recognition inference (`None` = auto).
        max_workers: Maximum worker thread count clamp for recognition inference (`None` = engine default 8).
    """
    def __init__(
        self,
        *,
        batch_size: int | None = None,
        workers: int | None = None,
        max_workers: int | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            workers=workers,
            max_workers=max_workers,
        )


class PostRecognitionOptions(_GeneratedPostRecognitionOptions):
    """Phase 6: Post-recognition filtering options."""
    def __init__(
        self,
        *,
        score_threshold: float | None = None,
    ):
        super().__init__(
            score_threshold=score_threshold,
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
    """Text recognition model configuration.

    Args:
        enabled: Whether recognition stage is enabled.
        model_path: Filesystem path to recognition model (.rten or .onnx).
        dict_path: Filesystem path to dictionary vocabulary file (.txt).
        rec_batch_num: Recognition batch size. Default: 1.
        rec_workers: Number of parallel worker threads for recognition (`None` = auto).
        max_rec_workers: Maximum worker thread count clamp (`None` = engine default 8).
    """
    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        dict_path: str | None = None,
        rec_batch_num: int | None = None,
        rec_workers: int | None = None,
        max_rec_workers: int | None = None,
    ):
        super().__init__(
            enabled=enabled,
            model_path=model_path,
            dict_path=dict_path,
            rec_batch_num=rec_batch_num,
            rec_workers=rec_workers,
            max_rec_workers=max_rec_workers,
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

    @classmethod
    def _from_generated(cls, raw: _GeneratedInitializeConfig) -> InitializeConfig:
        return cls(
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

    @classmethod
    def indra(
        cls,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        dict_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a preset configuration for Indra OCR."""
        if det_model_path and rec_model_path and dict_path:
            return cls._from_generated(create_initialize_config_indra(det_model_path, rec_model_path, dict_path))
        return cls(
            model="indraocr",
            template="indra",
            detection=DetectionConfig(model_path=det_model_path) if det_model_path else None,
            recognition=RecognitionConfig(model_path=rec_model_path, dict_path=dict_path)
            if (rec_model_path or dict_path)
            else None,
        )

    @classmethod
    def ppv6(
        cls,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        dict_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a preset configuration for PP-OCRv6."""
        if det_model_path and rec_model_path and dict_path:
            return cls._from_generated(create_initialize_config_ppv6(det_model_path, rec_model_path, dict_path))
        return cls(
            model="ppocrv6-tiny",
            template="ppv6",
            detection=DetectionConfig(model_path=det_model_path) if det_model_path else None,
            recognition=RecognitionConfig(model_path=rec_model_path, dict_path=dict_path)
            if (rec_model_path or dict_path)
            else None,
        )

    @classmethod
    def ppv5(
        cls,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        dict_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a preset configuration for PP-OCRv5."""
        return cls(
            model="ppocrv5",
            template="ppv5",
            detection=DetectionConfig(model_path=det_model_path) if det_model_path else None,
            recognition=RecognitionConfig(model_path=rec_model_path, dict_path=dict_path)
            if (rec_model_path or dict_path)
            else None,
        )

    @classmethod
    def ppv4(
        cls,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        dict_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a preset configuration for PP-OCRv4."""
        return cls(
            model="ppocrv4",
            template="ppv4",
            detection=DetectionConfig(model_path=det_model_path) if det_model_path else None,
            recognition=RecognitionConfig(model_path=rec_model_path, dict_path=dict_path)
            if (rec_model_path or dict_path)
            else None,
        )

    @classmethod
    def ppv3(
        cls,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        dict_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a preset configuration for PP-OCRv3."""
        return cls(
            template="ppv3",
            detection=DetectionConfig(model_path=det_model_path) if det_model_path else None,
            recognition=RecognitionConfig(model_path=rec_model_path, dict_path=dict_path)
            if (rec_model_path or dict_path)
            else None,
        )

    @classmethod
    def for_model(
        cls,
        model: str | None = "ppocrv6-tiny",
        lang: str | None = None,
        model_path: str | None = None,
    ) -> InitializeConfig:
        """Creates a configuration for a canonical model identifier."""
        return create_initialize_config_for_model(model=model, lang=lang, model_path=model_path)


class OcrRunOptions(_GeneratedOcrRunOptions):
    """Per-request execution options for text detection."""
    def __init__(
        self,
        output: OutputGranularity = OutputGranularity.LINES,
        auto_unwarp: bool | None = None,
        layout_reading: bool | None = None,
        layout_model: str | None = None,
        orientation_detection: bool | None = None,
        line_classification: bool | None = None,
        pre_detection: PreDetectionOptions | None = None,
        detection: DetectionRunOptions | None = None,
        post_detection: PostDetectionOptions | None = None,
        pre_recognition: PreRecognitionOptions | None = None,
        recognition: RecognitionRunOptions | None = None,
        post_recognition: PostRecognitionOptions | None = None,
    ):
        super().__init__(
            output=output,
            auto_unwarp=auto_unwarp,
            layout_reading=layout_reading,
            layout_model=layout_model,
            orientation_detection=orientation_detection,
            line_classification=line_classification,
            pre_detection=pre_detection,
            detection=detection,
            post_detection=post_detection,
            pre_recognition=pre_recognition,
            recognition=recognition,
            post_recognition=post_recognition,
        )


class DetectTextBuilder:
    """Fluent builder for chaining text detection options and running inference."""
    def __init__(self, engine: RustO, source: ImageSource | str | os.PathLike[str] | bytes | None = None):
        self._engine = engine
        self._source = self._normalize_source(source) if source is not None else None
        self._rgb: tuple[bytes, int, int, int] | None = None
        self._output: OutputGranularity = OutputGranularity.LINES
        self._auto_unwarp: bool | None = None
        self._layout_reading: bool | None = None
        self._layout_model: str | None = None
        self._orientation_detection: bool | None = None
        self._line_classification: bool | None = None
        self._pre_detection: PreDetectionOptions | None = None
        self._detection: DetectionRunOptions | None = None
        self._post_detection: PostDetectionOptions | None = None
        self._pre_recognition: PreRecognitionOptions | None = None
        self._recognition: RecognitionRunOptions | None = None
        self._post_recognition: PostRecognitionOptions | None = None

    @staticmethod
    def _normalize_source(source: ImageSource | str | os.PathLike[str] | bytes | bytearray) -> ImageSource:
        if isinstance(source, ImageSource):
            return source
        if isinstance(source, (str, os.PathLike)):
            return ImageSource.PATH(os.fspath(source))
        if isinstance(source, (bytes, bytearray)):
            return ImageSource.BYTES(bytes(source))
        raise TypeError(f"Invalid image source: {source}")

    def image(self, source: ImageSource | str | os.PathLike[str] | bytes | bytearray) -> DetectTextBuilder:
        """Specify the image source."""
        self._source = self._normalize_source(source)
        self._rgb = None
        return self

    def data(self, data: bytes | bytearray) -> DetectTextBuilder:
        """Specify encoded image bytes."""
        self._source = ImageSource.BYTES(bytes(data))
        self._rgb = None
        return self

    def rgb(
        self,
        pixels: bytes | bytearray,
        width: int,
        height: int,
        stride: int | None = None,
    ) -> DetectTextBuilder:
        """Specify raw decoded 24-bit interleaved RGB pixels."""
        effective_stride = stride if stride is not None else width * 3
        self._rgb = (bytes(pixels), width, height, effective_stride)
        self._source = None
        return self

    def with_auto_unwarp(self, enabled: bool = True) -> DetectTextBuilder:
        """Enable or disable document unwarping."""
        self._auto_unwarp = enabled
        return self

    def with_layout_reading(self, enabled: bool = True) -> DetectTextBuilder:
        """Enable or disable document layout reading order analysis."""
        self._layout_reading = enabled
        return self

    def with_layout_model(self, model: str) -> DetectTextBuilder:
        """Set layout model identifier."""
        self._layout_model = model
        return self

    def with_orientation_detection(self, enabled: bool = True) -> DetectTextBuilder:
        """Enable or disable document orientation detection."""
        self._orientation_detection = enabled
        return self

    def with_line_classification(self, enabled: bool = True) -> DetectTextBuilder:
        """Enable or disable text line orientation classification."""
        self._line_classification = enabled
        return self

    def with_pre_detection(
        self,
        options: PreDetectionOptions | None = None,
        *,
        descreen_strength: float | None = None,
        target_max_side: int | None = None,
        crop_padding_x: float | None = None,
        crop_padding_y: float | None = None,
    ) -> DetectTextBuilder:
        """Configure pre-detection options."""
        if options is not None:
            self._pre_detection = options
        else:
            self._pre_detection = PreDetectionOptions(
                descreen_strength=descreen_strength,
                target_max_side=target_max_side,
                crop_padding_x=crop_padding_x,
                crop_padding_y=crop_padding_y,
            )
        return self

    def with_detection(
        self,
        options: DetectionRunOptions | None = None,
        *,
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> DetectTextBuilder:
        """Configure detection inference options."""
        if options is not None:
            self._detection = options
        else:
            self._detection = DetectionRunOptions(
                limit_side_len=limit_side_len,
                limit_type=limit_type,
                mean=mean,
                std=std,
            )
        return self

    def with_post_detection(
        self,
        options: PostDetectionOptions | None = None,
        *,
        threshold: float | None = None,
        box_threshold: float | None = None,
        unclip_ratio: float | None = None,
        use_dilation: bool | None = None,
        score_mode: str | None = None,
        max_candidates: int | None = None,
        min_height: float | None = None,
        min_side_len: float | None = None,
        max_side_len: float | None = None,
        width_height_ratio: float | None = None,
    ) -> DetectTextBuilder:
        """Configure post-detection filtering options."""
        if options is not None:
            self._post_detection = options
        else:
            self._post_detection = PostDetectionOptions(
                threshold=threshold,
                box_threshold=box_threshold,
                unclip_ratio=unclip_ratio,
                use_dilation=use_dilation,
                score_mode=score_mode,
                max_candidates=max_candidates,
                min_height=min_height,
                min_side_len=min_side_len,
                max_side_len=max_side_len,
                width_height_ratio=width_height_ratio,
            )
        return self

    def with_pre_recognition(
        self,
        options: PreRecognitionOptions | None = None,
        *,
        line_y_threshold: float | None = None,
        word_x_threshold: float | None = None,
    ) -> DetectTextBuilder:
        """Configure pre-recognition clustering options."""
        if options is not None:
            self._pre_recognition = options
        else:
            self._pre_recognition = PreRecognitionOptions(
                line_y_threshold=line_y_threshold,
                word_x_threshold=word_x_threshold,
            )
        return self

    def with_recognition(
        self,
        options: RecognitionRunOptions | None = None,
        *,
        batch_size: int | None = None,
        workers: int | None = None,
        max_workers: int | None = None,
    ) -> DetectTextBuilder:
        """Configure recognition inference options."""
        if options is not None:
            self._recognition = options
        else:
            self._recognition = RecognitionRunOptions(
                batch_size=batch_size,
                workers=workers,
                max_workers=max_workers,
            )
        return self

    def with_post_recognition(
        self,
        options: PostRecognitionOptions | None = None,
        *,
        score_threshold: float | None = None,
    ) -> DetectTextBuilder:
        """Configure post-recognition filtering options."""
        if options is not None:
            self._post_recognition = options
        else:
            self._post_recognition = PostRecognitionOptions(score_threshold=score_threshold)
        return self

    def as_lines(self) -> DetectTextBuilder:
        """Set output format to line-level results."""
        self._output = OutputGranularity.LINES
        return self

    def as_words(self) -> DetectTextBuilder:
        """Set output format to word-level results."""
        self._output = OutputGranularity.WORDS
        return self

    def as_spatial(self) -> DetectTextBuilder:
        """Set output format to 2D spatial text layout."""
        self._output = OutputGranularity.SPATIAL
        return self

    def as_tsv(self) -> DetectTextBuilder:
        """Set output format to TSV (tab-separated values)."""
        self._output = OutputGranularity.TSV
        return self

    def with_output(self, granularity: OutputGranularity) -> DetectTextBuilder:
        """Set output granularity."""
        self._output = granularity
        return self

    def with_options(self, options: OcrRunOptions) -> DetectTextBuilder:
        """Merge options from an OcrRunOptions instance."""
        self._output = options.output
        self._auto_unwarp = options.auto_unwarp
        self._layout_reading = options.layout_reading
        self._layout_model = options.layout_model
        self._orientation_detection = options.orientation_detection
        self._line_classification = options.line_classification
        self._pre_detection = options.pre_detection
        self._detection = options.detection
        self._post_detection = options.post_detection
        self._pre_recognition = options.pre_recognition
        self._recognition = options.recognition
        self._post_recognition = options.post_recognition
        return self

    def run(self, source: ImageSource | str | os.PathLike[str] | bytes | None = None) -> DetectTextResult:
        """Execute OCR detection with chained options."""
        options = OcrRunOptions(
            output=self._output,
            auto_unwarp=self._auto_unwarp,
            layout_reading=self._layout_reading,
            layout_model=self._layout_model,
            orientation_detection=self._orientation_detection,
            line_classification=self._line_classification,
            pre_detection=self._pre_detection,
            detection=self._detection,
            post_detection=self._post_detection,
            pre_recognition=self._pre_recognition,
            recognition=self._recognition,
            post_recognition=self._post_recognition,
        )
        if source is not None:
            return self._engine._raw_detect_text(self._normalize_source(source), options)
        if self._rgb is not None:
            pixels, width, height, stride = self._rgb
            return self._engine.detect_text_rgb(pixels, width, height, stride, options)
        if self._source is None:
            raise ValueError("Image source must be provided to detect_text() or run()")
        return self._engine._raw_detect_text(self._source, options)


_original_initialize = RustO.initialize
_original_detect_text = RustO.detect_text
_original_detect_text_rgb = RustO.detect_text_rgb


def _raw_detect_text(self, source, options):
    if getattr(self, "_handle", None) is None:
        raise RuntimeError("RustO engine is closed")
    return _original_detect_text(self, source, options)


def _detect_text_rgb(self, pixels, width, height, stride, options=None):
    if getattr(self, "_handle", None) is None:
        raise RuntimeError("RustO engine is closed")
    return _original_detect_text_rgb(self, pixels, width, height, stride, options)


RustO._raw_detect_text = _raw_detect_text


def _initialize(config: InitializeConfig | str | None = None) -> RustO:
    if config is None:
        return _original_initialize(InitializeConfig())
    if isinstance(config, str):
        return RustO.initialize_from_json(config)
    return _original_initialize(config)


def _detect_text(self, source=None, options=None):
    if source is None:
        return DetectTextBuilder(self)
    if options is None:
        return DetectTextBuilder(self, source)
    norm_source = DetectTextBuilder._normalize_source(source)
    return self._raw_detect_text(norm_source, options)


def _detect_text_builder(self, source=None) -> DetectTextBuilder:
    return DetectTextBuilder(self, source)


def _close(self) -> None:
    handle = getattr(self, "_handle", None)
    if handle is not None:
        self._handle = None
        _uniffi_rust_call(_UniffiLib.uniffi_rusto_fn_free_rusto, handle)


def _enter(self) -> RustO:
    return self


def _exit(self, exc_type, exc_val, exc_tb) -> None:
    self.close()


RustO.initialize = staticmethod(_initialize)
RustO.detect_text = _detect_text
RustO.detect_text_rgb = _detect_text_rgb
RustO.detect_text_builder = _detect_text_builder
RustO.close = _close
RustO.__enter__ = _enter
RustO.__exit__ = _exit


def _is_ready(self) -> bool:
    handle = getattr(self, "_handle", None)
    return handle is not None


def _version(self) -> str:
    if getattr(self, "_handle", None) is None:
        raise RuntimeError("RustO engine is closed")
    return rusto_version()


RustO.is_ready = _is_ready
RustO.version = _version


def create_initialize_config_for_model(
    model: str | None = None,
    lang: str | None = None,
    model_path: str | None = None,
) -> InitializeConfig:
    """Creates an InitializeConfig for a canonical model identifier and optional model directory."""
    raw = _generated_create_initialize_config_for_model(model, lang, model_path)
    return InitializeConfig._from_generated(raw)


def download_model(
    model: str | None = None,
    lang: str | None = None,
    cache_dir: str | None = None,
) -> list[str]:
    """Downloads model assets for a specific OCR pipeline ahead of time."""
    return _generated_download_model(model, lang, cache_dir)


def version() -> str:
    """Returns the underlying RustO engine version string."""
    return rusto_version()


def is_ready(engine: RustO | None = None) -> bool:
    """Checks whether an engine instance is ready for inference."""
    if engine is None:
        return True
    return engine.is_ready()


RustO.download_model = staticmethod(download_model)


__all__ = [
    "RustO",
    "RustOProtocol",
    "DetectTextBuilder",
    "InitializeConfig",
    "DetectionConfig",
    "RecognitionConfig",
    "ClassificationConfig",
    "OrientationConfig",
    "UnwarpConfig",
    "LayoutConfig",
    "LayoutType",
    "OcrRunOptions",
    "PreDetectionOptions",
    "DetectionRunOptions",
    "PostDetectionOptions",
    "PreRecognitionOptions",
    "RecognitionRunOptions",
    "PostRecognitionOptions",
    "Frame",
    "RustoBindingError",
    "ImageSource",
    "OutputGranularity",
    "TextResult",
    "DetectTextResult",
    "default_ocr_run_options",
    "create_initialize_config_indra",
    "create_initialize_config_ppv6",
    "create_initialize_config_for_model",
    "download_model",
    "version",
    "is_ready",
    "rusto_version",
]

