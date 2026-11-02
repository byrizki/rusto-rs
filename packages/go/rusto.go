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

// LayoutRegion represents a detected document layout region.
type LayoutRegion struct {
	// XMin is the left coordinate of the bounding box.
	XMin int `json:"x_min"`
	// YMin is the top coordinate of the bounding box.
	YMin int `json:"y_min"`
	// XMax is the right coordinate of the bounding box.
	XMax int `json:"x_max"`
	// YMax is the bottom coordinate of the bounding box.
	YMax int `json:"y_max"`
	// LayoutType is the category identifier (e.g. 0=Text, 1=Title, 2=Figure, etc.).
	LayoutType int `json:"layout_type"`
	// Confidence is the detection confidence score between 0.0 and 1.0.
	Confidence float32 `json:"confidence"`
}

// Engine wraps an active RustO OCR instance with thread-safe synchronization.
type Engine struct {
	mu     sync.RWMutex
	handle *C.ROCRHandle
}

// New creates and initializes a new OCR engine instance using JSON configuration.
func New(configJSON string) (*Engine, error) {
	cConfig := C.CString(configJSON)
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

// DetectTextFile detects text from an image file at the given path.
func (e *Engine) DetectTextFile(path string, optionsJSON string) ([]TextResult, error) {
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

// DetectTextData detects text from raw in-memory image bytes.
func (e *Engine) DetectTextData(data []byte, optionsJSON string) ([]TextResult, error) {
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

// DetectTextFileSpatial exports spatial text layout from an image file.
func (e *Engine) DetectTextFileSpatial(path string, optionsJSON string) (string, error) {
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

// DetectTextDataSpatial exports spatial text layout from raw in-memory image bytes.
func (e *Engine) DetectTextDataSpatial(data []byte, optionsJSON string) (string, error) {
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

// DetectTextRGB detects text from raw decoded interleaved RGB8 pixel bytes.
func (e *Engine) DetectTextRGB(pixels []byte, width, height uint32, stride uint, optionsJSON string) ([]TextResult, error) {
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

// DetectTextRGBSpatial exports spatial text layout from raw decoded interleaved RGB8 pixel bytes.
func (e *Engine) DetectTextRGBSpatial(pixels []byte, width, height uint32, stride uint, optionsJSON string) (string, error) {
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

// DetectLayoutFile detects document layout regions from an image file at the given path.
func (e *Engine) DetectLayoutFile(path string) ([]LayoutRegion, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return nil, ErrEngineClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cRegions *C.CLayoutRegion
	var count C.size_t

	ret := C.rocr_detect_layout_file(e.handle, cPath, &cRegions, &count)
	if ret != 0 {
		return nil, ErrDetectionFailed
	}
	defer C.rocr_free_layout_regions(cRegions, count)

	return convertLayoutRegions(cRegions, count), nil
}

// DetectLayoutRGB detects document layout regions from raw decoded interleaved RGB8 pixel bytes.
func (e *Engine) DetectLayoutRGB(pixels []byte, width, height uint32, stride uint) ([]LayoutRegion, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.handle == nil {
		return nil, ErrEngineClosed
	}
	if len(pixels) == 0 {
		return []LayoutRegion{}, nil
	}

	var cRegions *C.CLayoutRegion
	var count C.size_t

	ret := C.rocr_detect_layout_rgb(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&pixels[0])),
		C.size_t(len(pixels)),
		C.uint32_t(width),
		C.uint32_t(height),
		C.size_t(stride),
		&cRegions,
		&count,
	)
	if ret != 0 {
		return nil, ErrDetectionFailed
	}
	defer C.rocr_free_layout_regions(cRegions, count)

	return convertLayoutRegions(cRegions, count), nil
}

func convertLayoutRegions(cRegions *C.CLayoutRegion, count C.size_t) []LayoutRegion {
	if cRegions == nil || count == 0 {
		return []LayoutRegion{}
	}

	regionsSlice := (*[1 << 28]C.CLayoutRegion)(unsafe.Pointer(cRegions))[:count:count]
	out := make([]LayoutRegion, count)

	for i := 0; i < int(count); i++ {
		r := regionsSlice[i]
		out[i] = LayoutRegion{
			XMin:       int(r.x_min),
			YMin:       int(r.y_min),
			XMax:       int(r.x_max),
			YMax:       int(r.y_max),
			LayoutType: int(r.layout_type),
			Confidence: float32(r.confidence),
		}
	}
	return out
}
