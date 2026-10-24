package rusto

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR}/../../target/debug -L${SRCDIR}/../../target/release -lrusto
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
	ErrInitFailed       = errors.New("rusto: failed to initialize OCR engine")
	ErrDetectionFailed  = errors.New("rusto: text detection failed")
	ErrEngineClosed     = errors.New("rusto: engine handle is closed")
)

// Frame represents bounding frame dimensions.
type Frame struct {
	Width  float32 `json:"width"`
	Height float32 `json:"height"`
	Top    float32 `json:"top"`
	Left   float32 `json:"left"`
}

// TextResult contains recognized text, confidence score, and bounding coordinates.
type TextResult struct {
	Text      string       `json:"text"`
	Score     float32      `json:"score"`
	BoxPoints [4][2]float32 `json:"box_points"`
	Frame     Frame        `json:"frame"`
}

// Engine wraps an active RustO OCR instance.
type Engine struct {
	mu     sync.RWMutex
	handle C.ROCRHandle
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

func (e *Engine) convertResults(resultsPtr unsafe.Pointer, count C.size_t) []TextResult {
	if resultsPtr == nil || count == 0 {
		return []TextResult{}
	}

	cResults := (*[1 << 28]C.CTextResult)(resultsPtr)[:count:count]
	out := make([]TextResult, count)

	for i := 0; i < int(count); i++ {
		res := cResults[i]
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

	var results unsafe.Pointer
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

	var results unsafe.Pointer
	var count C.size_t

	status := C.rocr_detect_text_data(
		e.handle,
		unsafe.Pointer(&data[0]),
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
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
		cOptions,
	)
	if cText == nil {
		return "", ErrDetectionFailed
	}
	defer C.rocr_free_string(cText)

	return C.GoString(cText), nil
}
