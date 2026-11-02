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

func TestDetectIndraOCRIdCard(t *testing.T) {
	detPath := "../../models/indra/det.rten"
	recPath := "../../models/indra/rec.rten"
	dictPath := "../../models/indra/meta.json"
	imgPath := "../../models/test_images/idcard.jpg"

	if _, err := os.Stat(detPath); os.IsNotExist(err) {
		t.Skip("Indra model files not found, skipping real OCR test")
	}

	config := fmt.Sprintf(`{
		"detection": {"model_path": "%s"},
		"recognition": {"model_path": "%s", "dict_path": "%s"}
	}`, detPath, recPath, dictPath)

	engine, err := New(config)
	if err != nil {
		t.Fatalf("Failed to initialize engine with Indra OCR: %v", err)
	}
	defer engine.Close()

	results, err := engine.DetectTextFile(imgPath, "{}")
	if err != nil {
		t.Fatalf("DetectTextFile with Indra OCR failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("Expected non-empty text results on idcard.jpg")
	}

	foundPassport := false
	for _, res := range results {
		if strings.Contains(res.Text, "PASSPORT") || strings.Contains(res.Text, "UNITED") || strings.Contains(res.Text, "BRITAIN") {
			foundPassport = true
			break
		}
	}
	if !foundPassport {
		t.Fatalf("Expected detected text to include passport keywords, got %v", results)
	}
}

func TestDetectLayoutAPI(t *testing.T) {
	detPath := "../../models/PPOCR_v6/det.rten"
	recPath := "../../models/PPOCR_v6/rec.rten"
	dictPath := "../../models/PPOCR_v6/dict.txt"

	if _, err := os.Stat(detPath); os.IsNotExist(err) {
		t.Skip("Model files not found, skipping layout test")
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

	// Calling DetectLayoutFile without layout model configured should return an error
	_, err = engine.DetectLayoutFile("../../models/test_images/example1.png")
	if err == nil {
		t.Fatal("expected error when layout model is not configured, got nil")
	}

	// Calling DetectLayoutRGB without layout model configured should return an error
	pixels := make([]byte, 100*100*3)
	_, err = engine.DetectLayoutRGB(pixels, 100, 100, 300)
	if err == nil {
		t.Fatal("expected error when layout model is not configured, got nil")
	}
}

func TestDetectChronamDensePageWithLayout(t *testing.T) {
	detPath := "../../models/PPOCR_v6_tiny/det.rten"
	recPath := "../../models/PPOCR_v6_tiny/rec.rten"
	dictPath := "../../models/PPOCR_v6_tiny/dict.txt"
	layoutPath := "../../models/PPOCR_v6_tiny/layout_cdla.onnx"
	imgPath := "../../tests/fixtures/chronam-ad-page.png"

	if _, err := os.Stat(detPath); os.IsNotExist(err) {
		t.Skip("Model files not found, skipping dense page test")
	}
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping dense page test")
	}

	config := fmt.Sprintf(`{
		"detection": {"modelPath": "%s", "enabled": true},
		"recognition": {"modelPath": "%s", "dictPath": "%s", "enabled": true},
		"layout": {"modelPath": "%s", "enabled": true}
	}`, detPath, recPath, dictPath, layoutPath)

	engine, err := New(config)
	if err != nil {
		t.Fatalf("Failed to initialize engine with layout: %v", err)
	}
	defer engine.Close()

	results, err := engine.DetectTextFile(imgPath, `{"layout": true}`)
	if err != nil {
		t.Fatalf("DetectTextFile with layout failed: %v", err)
	}

	if len(results) != 153 {
		t.Fatalf("Expected exactly 153 lines, got %d", len(results))
	}

	if results[0].Text != "All Orders in our" {
		t.Fatalf("Expected first line to be 'All Orders in our', got '%s'", results[0].Text)
	}
	if results[0].RegionID < 0 {
		t.Fatalf("Expected RegionID >= 0 when layout is enabled, got %d", results[0].RegionID)
	}
	if results[0].LayoutType < 0 {
		t.Fatalf("Expected LayoutType >= 0 when layout is enabled, got %d", results[0].LayoutType)
	}

	regions, err := engine.DetectLayoutFile(imgPath)
	if err != nil {
		t.Fatalf("DetectLayoutFile failed: %v", err)
	}
	if len(regions) < 1 {
		t.Fatalf("Expected at least 1 layout region, got %d", len(regions))
	}
}


