// Package rusto provides high-performance, pure-Rust OCR engine bindings for Golang via Cgo.
package rusto

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../target/debug -L${SRCDIR}/../../target/release -Wl,-rpath,${SRCDIR}/../../target/release -Wl,-rpath,${SRCDIR}/../../target/debug -lrusto
#include <stdlib.h>
#include "rusto.h"
*/
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

var (
	// ErrInitFailed indicates that native engine initialization failed.
	ErrInitFailed = errors.New("rusto: failed to initialize OCR engine")
	// ErrDetectionFailed indicates that native text detection failed.
	ErrDetectionFailed = errors.New("rusto: text detection failed")
	// ErrEngineClosed indicates an operation was attempted on a closed engine instance.
	ErrEngineClosed = errors.New("rusto: engine handle is closed")
)

// Frame represents an axis-aligned bounding box frame with position and dimensions in pixels.
type Frame struct {
	// Width of the bounding box in pixels.
	Width float32 `json:"width"`
	// Height of the bounding box in pixels.
	Height float32 `json:"height"`
	// Top vertical coordinate (Y) of the bounding box.
	Top float32 `json:"top"`
	// Left horizontal coordinate (X) of the bounding box.
	Left float32 `json:"left"`
}

// TextResult contains recognized text, confidence score, bounding polygon coordinates, and axis-aligned frame.
type TextResult struct {
	// Text is the recognized text string.
	Text string `json:"text"`
	// Score is the recognition confidence score between 0.0 and 1.0.
	Score float32 `json:"score"`
	// BoxPoints contains 4 vertex coordinates [top-left, top-right, bottom-right, bottom-left].
	BoxPoints [4][2]float32 `json:"box_points"`
	// Frame is the axis-aligned bounding box enclosing the text.
	Frame Frame `json:"frame"`
	// RegionID is the detected layout region index, or -1 if unassigned.
	RegionID int32 `json:"region_id"`
	// LayoutType is the semantic layout category identifier, or -1 if unassigned.
	LayoutType int32 `json:"layout_type"`
}

// DetectTextResult holds either structured recognition items or a reconstructed 2D spatial layout string.
type DetectTextResult struct {
	// Results contains structured lines or words when IsSpatial is false.
	Results []TextResult `json:"results,omitempty"`
	// Text contains the formatted 2D spatial layout string when IsSpatial is true.
	Text string `json:"text,omitempty"`
	// IsSpatial indicates whether this result was produced with AsSpatial().
	IsSpatial bool `json:"is_spatial"`
}

// Engine wraps an active RustO OCR instance with thread-safe synchronization.
type Engine struct {
	mu     sync.RWMutex
	handle *C.ROCRHandle
}

// RustO is an alias to Engine representing an initialized OCR pipeline instance.
type RustO = Engine

// Initialize creates and initializes a new OCR engine instance using optional JSON configuration.
// When omitted, defaults to "{}" which initializes PP-OCRv6 Tiny FP32 (auto-downloading models if needed).
func Initialize(configJSON ...string) (*RustO, error) {
	return New(configJSON...)
}

// New creates and initializes a new OCR engine instance using optional JSON configuration.
func New(configJSON ...string) (*Engine, error) {
	cfg := "{}"
	if len(configJSON) > 0 {
		cfg = configJSON[0]
	}

	cConfig := C.CString(cfg)
	defer C.free(unsafe.Pointer(cConfig))

	handle := C.rocr_initialize(cConfig)
	if handle == nil {
		return nil, ErrInitFailed
	}

	eng := &Engine{handle: handle}
	runtime.SetFinalizer(eng, func(e *Engine) {
		e.Close()
	})
	return eng, nil
}

// Close releases the native OCR engine resources.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.handle != nil {
		C.rocr_free(e.handle)
		e.handle = nil
	}
}

// Version returns the engine library version string.
func Version() string {
	cV := C.rocr_version()
	if cV == nil {
		return "0.4.0"
	}
	return C.GoString(cV)
}

// Version returns the engine library version string.
func (e *Engine) Version() string {
	return Version()
}

// IsReady returns true if the engine handle is initialized and ready for inference.
func (e *Engine) IsReady() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.handle != nil && C.rocr_is_ready(e.handle) != 0
}

// DownloadModel downloads model assets for a specific OCR pipeline ahead of time.
func DownloadModel(model, lang, cacheDir string) ([]string, error) {
	var cModel, cLang, cCacheDir *C.char
	if model != "" {
		cModel = C.CString(model)
		defer C.free(unsafe.Pointer(cModel))
	}
	if lang != "" {
		cLang = C.CString(lang)
		defer C.free(unsafe.Pointer(cLang))
	}
	if cacheDir != "" {
		cCacheDir = C.CString(cacheDir)
		defer C.free(unsafe.Pointer(cCacheDir))
	}

	cJson := C.rocr_download_model(cModel, cLang, cCacheDir)
	if cJson == nil {
		return nil, errors.New("rusto: failed to download model")
	}
	defer C.rocr_free_string(cJson)

	var paths []string
	if err := json.Unmarshal([]byte(C.GoString(cJson)), &paths); err != nil {
		return nil, fmt.Errorf("rusto: failed to parse downloaded paths: %w", err)
	}
	return paths, nil
}


func (e *Engine) convertResults(cResults *C.CTextResult, count C.size_t) []TextResult {
	if cResults == nil || count == 0 {
		return []TextResult{}
	}

	resultsSlice := (*[1 << 28]C.CTextResult)(unsafe.Pointer(cResults))[:count:count]
	out := make([]TextResult, count)

	for i := 0; i < int(count); i++ {
		res := resultsSlice[i]
		text := ""
		if res.text != nil {
			text = C.GoString(res.text)
		}

		out[i] = TextResult{
			Text:  text,
			Score: float32(res.score),
			BoxPoints: [4][2]float32{
				{float32(res.box_x1), float32(res.box_y1)},
				{float32(res.box_x2), float32(res.box_y2)},
				{float32(res.box_x3), float32(res.box_y3)},
				{float32(res.box_x4), float32(res.box_y4)},
			},
			Frame: Frame{
				Width:  float32(res.frame_width),
				Height: float32(res.frame_height),
				Top:    float32(res.frame_top),
				Left:   float32(res.frame_left),
			},
			RegionID:   int32(res.region_id),
			LayoutType: int32(res.layout_type),
		}
	}
	return out
}

func (e *Engine) detectTextFile(path string, optionsJSON string) ([]TextResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return nil, ErrEngineClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	var results *C.CTextResult
	var count C.size_t

	status := C.rocr_detect_text_file(e.handle, cPath, cOptions, &results, &count)
	if status != 0 {
		return nil, fmt.Errorf("%w: status code %d", ErrDetectionFailed, int(status))
	}
	defer C.rocr_free_results(results, count)

	return e.convertResults(results, count), nil
}

func (e *Engine) detectTextData(data []byte, optionsJSON string) ([]TextResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return nil, ErrEngineClosed
	}
	if len(data) == 0 {
		return []TextResult{}, nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	var results *C.CTextResult
	var count C.size_t

	status := C.rocr_detect_text_data(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		cOptions,
		&results,
		&count,
	)
	if status != 0 {
		return nil, fmt.Errorf("%w: status code %d", ErrDetectionFailed, int(status))
	}
	defer C.rocr_free_results(results, count)

	return e.convertResults(results, count), nil
}

func (e *Engine) detectTextFileSpatial(path string, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_file_spatial(e.handle, cPath, cOptions)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

func (e *Engine) detectTextFileTsv(path string, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_file_tsv(e.handle, cPath, cOptions)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

func (e *Engine) detectTextDataSpatial(data []byte, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}
	if len(data) == 0 {
		return "", nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_data_spatial(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		cOptions,
	)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

func (e *Engine) detectTextDataTsv(data []byte, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}
	if len(data) == 0 {
		return "", nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_data_tsv(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		cOptions,
	)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

func (e *Engine) detectTextRGB(pixels []byte, width, height uint32, stride uint, optionsJSON string) ([]TextResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return nil, ErrEngineClosed
	}
	if len(pixels) == 0 {
		return []TextResult{}, nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	var results *C.CTextResult
	var count C.size_t

	status := C.rocr_detect_text_rgb(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&pixels[0])),
		C.size_t(len(pixels)),
		C.uint32_t(width),
		C.uint32_t(height),
		C.size_t(stride),
		cOptions,
		&results,
		&count,
	)
	if status != 0 {
		return nil, fmt.Errorf("%w: status code %d", ErrDetectionFailed, int(status))
	}
	defer C.rocr_free_results(results, count)

	return e.convertResults(results, count), nil
}

func (e *Engine) detectTextRGBSpatial(pixels []byte, width, height uint32, stride uint, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}
	if len(pixels) == 0 {
		return "", nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_rgb_spatial(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&pixels[0])),
		C.size_t(len(pixels)),
		C.uint32_t(width),
		C.uint32_t(height),
		C.size_t(stride),
		cOptions,
	)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

func (e *Engine) detectTextRGBTsv(pixels []byte, width, height uint32, stride uint, optionsJSON string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return "", ErrEngineClosed
	}
	if len(pixels) == 0 {
		return "", nil
	}

	cOptions := C.CString(optionsJSON)
	defer C.free(unsafe.Pointer(cOptions))

	cText := C.rocr_detect_text_rgb_tsv(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&pixels[0])),
		C.size_t(len(pixels)),
		C.uint32_t(width),
		C.uint32_t(height),
		C.size_t(stride),
		cOptions,
	)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}

// PreDetectionOptions contains conditioning and downscale parameters.
type PreDetectionOptions struct {
	DescreenStrength *float32 `json:"descreenStrength,omitempty"`
	TargetMaxSide    *uint32  `json:"targetMaxSide,omitempty"`
	CropPaddingX     *float32 `json:"cropPaddingX,omitempty"`
	CropPaddingY     *float32 `json:"cropPaddingY,omitempty"`
}

// DetectionRunOptions contains DBNet detection parameters.
type DetectionRunOptions struct {
	LimitSideLen *int32      `json:"limitSideLen,omitempty"`
	LimitType    *string     `json:"limitType,omitempty"`
	Mean         *[3]float32 `json:"mean,omitempty"`
	Std          *[3]float32 `json:"std,omitempty"`
}

// PostDetectionOptions contains polygon filtering parameters.
type PostDetectionOptions struct {
	Threshold        *float32 `json:"threshold,omitempty"`
	BoxThreshold     *float32 `json:"boxThreshold,omitempty"`
	UnclipRatio      *float32 `json:"unclipRatio,omitempty"`
	UseDilation      *bool    `json:"useDilation,omitempty"`
	ScoreMode        *string  `json:"scoreMode,omitempty"`
	MaxCandidates    *int32   `json:"maxCandidates,omitempty"`
	MinHeight        *float32 `json:"minHeight,omitempty"`
	MinSideLen       *float32 `json:"minSideLen,omitempty"`
	MaxSideLen       *float32 `json:"maxSideLen,omitempty"`
	WidthHeightRatio *float32 `json:"widthHeightRatio,omitempty"`
}

// PreRecognitionOptions contains grouping tolerances.
type PreRecognitionOptions struct {
	LineYThreshold *float32 `json:"lineYThreshold,omitempty"`
	WordXThreshold *float32 `json:"wordXThreshold,omitempty"`
}

// RecognitionRunOptions contains recognition batching and worker concurrency parameters.
type RecognitionRunOptions struct {
	// BatchSize is the number of text crops processed concurrently.
	BatchSize *int32 `json:"batchSize,omitempty"`
	// Workers is the number of parallel worker threads for recognition inference (nil = auto).
	Workers *uint32 `json:"workers,omitempty"`
	// MaxWorkers is the maximum worker thread count clamp for recognition inference (nil = engine default 8).
	MaxWorkers *uint32 `json:"maxWorkers,omitempty"`
}

// PostRecognitionOptions contains recognition score filtering.
type PostRecognitionOptions struct {
	ScoreThreshold *float32 `json:"scoreThreshold,omitempty"`
}

// OcrRunOptions encapsulates all detection and recognition options across 6 phases and 4 toggles.
type OcrRunOptions struct {
	Output               string                  `json:"output,omitempty"`
	AutoUnwarp           *bool                   `json:"autoUnwarp,omitempty"`
	LayoutReading        *bool                   `json:"layoutReading,omitempty"`
	LayoutModel          *string                 `json:"layoutModel,omitempty"`
	OrientationDetection *bool                   `json:"orientationDetection,omitempty"`
	LineClassification   *bool                   `json:"lineClassification,omitempty"`
	PreDetection         *PreDetectionOptions    `json:"preDetection,omitempty"`
	Detection            *DetectionRunOptions    `json:"detection,omitempty"`
	PostDetection        *PostDetectionOptions   `json:"postDetection,omitempty"`
	PreRecognition       *PreRecognitionOptions  `json:"preRecognition,omitempty"`
	Recognition          *RecognitionRunOptions  `json:"recognition,omitempty"`
	PostRecognition      *PostRecognitionOptions `json:"postRecognition,omitempty"`
}

// DetectTextBuilder provides a fluent interface for chaining detection options.
type DetectTextBuilder struct {
	engine    *Engine
	filePath  string
	data      []byte
	hasData   bool
	pixels    []byte
	hasPixels bool
	width     uint32
	height    uint32
	stride    uint
	options   OcrRunOptions
}

// DetectText creates a fluent builder for chaining text detection options, with an optional image path.
func (e *Engine) DetectText(imagePath ...string) *DetectTextBuilder {
	b := &DetectTextBuilder{engine: e}
	if len(imagePath) > 0 && imagePath[0] != "" {
		b.filePath = imagePath[0]
	}
	return b
}

// DetectTextBuilder creates a fluent builder for chaining text detection options.
func (e *Engine) DetectTextBuilder() *DetectTextBuilder {
	return &DetectTextBuilder{engine: e}
}

// Image specifies the image file path.
func (b *DetectTextBuilder) Image(path string) *DetectTextBuilder {
	b.filePath = path
	b.hasData = false
	b.hasPixels = false
	return b
}

// Data specifies in-memory image bytes (e.g. PNG, JPEG).
func (b *DetectTextBuilder) Data(data []byte) *DetectTextBuilder {
	b.data = data
	b.hasData = true
	b.filePath = ""
	b.hasPixels = false
	return b
}

// RGB specifies raw interleaved RGB pixel bytes with dimensions.
func (b *DetectTextBuilder) RGB(pixels []byte, width, height uint32, stride uint) *DetectTextBuilder {
	b.pixels = pixels
	b.hasPixels = true
	b.width = width
	b.height = height
	if stride == 0 {
		b.stride = uint(width) * 3
	} else {
		b.stride = stride
	}
	b.filePath = ""
	b.hasData = false
	return b
}

// WithAutoUnwarp enables or disables document unwarping.
func (b *DetectTextBuilder) WithAutoUnwarp(enabled bool) *DetectTextBuilder {
	b.options.AutoUnwarp = &enabled
	return b
}

// WithLayoutReading enables or disables document layout reading order analysis.
func (b *DetectTextBuilder) WithLayoutReading(enabled bool) *DetectTextBuilder {
	b.options.LayoutReading = &enabled
	return b
}

// WithLayoutModel sets the document layout or reading direction model identifier.
func (b *DetectTextBuilder) WithLayoutModel(model string) *DetectTextBuilder {
	b.options.LayoutModel = &model
	return b
}

// WithOrientationDetection enables or disables document orientation detection.
func (b *DetectTextBuilder) WithOrientationDetection(enabled bool) *DetectTextBuilder {
	b.options.OrientationDetection = &enabled
	return b
}

// WithLineClassification enables or disables text line orientation classification.
func (b *DetectTextBuilder) WithLineClassification(enabled bool) *DetectTextBuilder {
	b.options.LineClassification = &enabled
	return b
}

// WithPreDetection configures phase 1 pre-detection options.
func (b *DetectTextBuilder) WithPreDetection(opts PreDetectionOptions) *DetectTextBuilder {
	b.options.PreDetection = &opts
	return b
}

// WithDetection configures phase 2 detection inference options.
func (b *DetectTextBuilder) WithDetection(opts DetectionRunOptions) *DetectTextBuilder {
	b.options.Detection = &opts
	return b
}

// WithPostDetection configures phase 3 post-detection options.
func (b *DetectTextBuilder) WithPostDetection(opts PostDetectionOptions) *DetectTextBuilder {
	b.options.PostDetection = &opts
	return b
}

// WithPreRecognition configures phase 4 pre-recognition options.
func (b *DetectTextBuilder) WithPreRecognition(opts PreRecognitionOptions) *DetectTextBuilder {
	b.options.PreRecognition = &opts
	return b
}

// WithRecognition configures phase 5 recognition inference options.
func (b *DetectTextBuilder) WithRecognition(opts RecognitionRunOptions) *DetectTextBuilder {
	b.options.Recognition = &opts
	return b
}

// WithPostRecognition configures phase 6 post-recognition options.
func (b *DetectTextBuilder) WithPostRecognition(opts PostRecognitionOptions) *DetectTextBuilder {
	b.options.PostRecognition = &opts
	return b
}

// AsLines sets output format to coherent text lines.
func (b *DetectTextBuilder) AsLines() *DetectTextBuilder {
	b.options.Output = "lines"
	return b
}

// AsWords sets output format to word-level tokens.
func (b *DetectTextBuilder) AsWords() *DetectTextBuilder {
	b.options.Output = "words"
	return b
}

// AsSpatial sets output format to 2D spatial layout.
func (b *DetectTextBuilder) AsSpatial() *DetectTextBuilder {
	b.options.Output = "spatial"
	return b
}

// AsTsv sets output format to TSV (tab-separated values).
func (b *DetectTextBuilder) AsTsv() *DetectTextBuilder {
	b.options.Output = "tsv"
	return b
}

// WithOutput sets output granularity ("lines", "words", or "spatial").
func (b *DetectTextBuilder) WithOutput(output string) *DetectTextBuilder {
	b.options.Output = output
	return b
}

// WithOptions merges options from an OcrRunOptions struct.
func (b *DetectTextBuilder) WithOptions(opts OcrRunOptions) *DetectTextBuilder {
	b.options = opts
	return b
}

// Run executes detection with chained options and returns a DetectTextResult.
func (b *DetectTextBuilder) Run() (*DetectTextResult, error) {
	bytes, err := json.Marshal(b.options)
	if err != nil {
		return nil, err
	}
	optsJSON := string(bytes)

	if b.options.Output == "spatial" {
		if b.filePath != "" {
			text, err := b.engine.detectTextFileSpatial(b.filePath, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		if b.hasData {
			text, err := b.engine.detectTextDataSpatial(b.data, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		if b.hasPixels {
			text, err := b.engine.detectTextRGBSpatial(b.pixels, b.width, b.height, b.stride, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		return nil, errors.New("rusto: no image source provided to DetectTextBuilder")
	}

	if b.options.Output == "tsv" {
		if b.filePath != "" {
			text, err := b.engine.detectTextFileTsv(b.filePath, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		if b.hasData {
			text, err := b.engine.detectTextDataTsv(b.data, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		if b.hasPixels {
			text, err := b.engine.detectTextRGBTsv(b.pixels, b.width, b.height, b.stride, optsJSON)
			if err != nil {
				return nil, err
			}
			return &DetectTextResult{Text: text, IsSpatial: true}, nil
		}
		return nil, errors.New("rusto: no image source provided to DetectTextBuilder")
	}

	if b.filePath != "" {
		items, err := b.engine.detectTextFile(b.filePath, optsJSON)
		if err != nil {
			return nil, err
		}
		return &DetectTextResult{Results: items, IsSpatial: false}, nil
	}
	if b.hasData {
		items, err := b.engine.detectTextData(b.data, optsJSON)
		if err != nil {
			return nil, err
		}
		return &DetectTextResult{Results: items, IsSpatial: false}, nil
	}
	if b.hasPixels {
		items, err := b.engine.detectTextRGB(b.pixels, b.width, b.height, b.stride, optsJSON)
		if err != nil {
			return nil, err
		}
		return &DetectTextResult{Results: items, IsSpatial: false}, nil
	}
	return nil, errors.New("rusto: no image source provided to DetectTextBuilder")
}
