package rusto

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestInitializationInvalidJSON(t *testing.T) {
	_, err := New("invalid-json-config")
	if err == nil {
		t.Fatal("expected error for invalid JSON configuration, got nil")
	}
	if !errors.Is(err, ErrInitFailed) {
		t.Fatalf("expected ErrInitFailed, got %v", err)
	}
}

func TestClosedEngineReturnsError(t *testing.T) {
	e := &Engine{handle: nil}

	_, err := e.DetectTextFile("/tmp/nonexistent.png", "{}")
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for file detect, got %v", err)
	}

	_, err = e.DetectTextData([]byte{0x01, 0x02}, "{}")
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for data detect, got %v", err)
	}

	_, err = e.DetectTextFileSpatial("/tmp/nonexistent.png", "{}")
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for spatial file detect, got %v", err)
	}

	_, err = e.DetectTextDataSpatial([]byte{0x01, 0x02}, "{}")
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for spatial data detect, got %v", err)
	}
}

func TestTextResultStructure(t *testing.T) {
	res := TextResult{
		Text:  "Sample OCR",
		Score: 0.99,
		BoxPoints: [4][2]float32{
			{0, 0},
			{100, 0},
			{100, 20},
			{0, 20},
		},
		Frame: Frame{
			Width:  100,
			Height: 20,
			Top:    0,
			Left:   0,
		},
	}

	if res.Text != "Sample OCR" {
		t.Fatalf("unexpected text: %s", res.Text)
	}
	if res.Score != 0.99 {
		t.Fatalf("unexpected score: %f", res.Score)
	}
	if res.Frame.Width != 100 {
		t.Fatalf("unexpected width: %f", res.Frame.Width)
	}
}

func TestDetectExample1Image(t *testing.T) {
	detPath := "../../models/PPOCR_v6/det.rten"
	recPath := "../../models/PPOCR_v6/rec.rten"
	dictPath := "../../models/PPOCR_v6/dict.txt"
	imgPath := "../../models/test_images/example1.png"

	if _, err := os.Stat(detPath); os.IsNotExist(err) {
		t.Skip("Model files not found, skipping real OCR test")
	}

	config := fmt.Sprintf(`{
		"detection": {"model_path": "%s"},
		"recognition": {"model_path": "%s", "dict_path": "%s"}
	}`, detPath, recPath, dictPath)

	engine, err := New(config)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer engine.Close()

	results, err := engine.DetectTextFile(imgPath, "{}")
	if err != nil {
		t.Fatalf("DetectTextFile failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("Expected non-empty text results on example1.png")
	}

	spatial, err := engine.DetectTextFileSpatial(imgPath, "{}")
	if err != nil {
		t.Fatalf("DetectTextFileSpatial failed: %v", err)
	}
	if !strings.Contains(spatial, "INVOICE") {
		t.Fatalf("Expected spatial text to contain 'INVOICE', got: %s", spatial)
	}
}

