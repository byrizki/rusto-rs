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

	_, err := e.DetectText("/tmp/nonexistent.png").AsLines().Run()
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for file detect, got %v", err)
	}

	_, err = e.DetectText().Data([]byte{0x01, 0x02}).AsLines().Run()
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for data detect, got %v", err)
	}

	_, err = e.DetectText("/tmp/nonexistent.png").AsSpatial().Run()
	if !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("expected ErrEngineClosed for spatial file detect, got %v", err)
	}

	_, err = e.DetectText().Data([]byte{0x01, 0x02}).AsSpatial().Run()
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

	structured, err := engine.DetectText(imgPath).AsLines().Run()
	if err != nil {
		t.Fatalf("DetectText AsLines failed: %v", err)
	}
	if structured.IsSpatial || len(structured.Results) == 0 {
		t.Fatalf("Expected non-empty structured text results on example1.png")
	}

	spatial, err := engine.DetectText(imgPath).AsSpatial().Run()
	if err != nil {
		t.Fatalf("DetectText AsSpatial failed: %v", err)
	}
	if !spatial.IsSpatial || !strings.Contains(spatial.Text, "INVOICE") {
		t.Fatalf("Expected spatial text to contain 'INVOICE', got: %s", spatial.Text)
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

	out, err := engine.DetectText(imgPath).AsLines().Run()
	if err != nil {
		t.Fatalf("DetectText with Indra OCR failed: %v", err)
	}
	if len(out.Results) == 0 {
		t.Fatalf("Expected non-empty text results on idcard.jpg")
	}

	foundPassport := false
	for _, res := range out.Results {
		if strings.Contains(res.Text, "PASSPORT") || strings.Contains(res.Text, "UNITED") || strings.Contains(res.Text, "BRITAIN") {
			foundPassport = true
			break
		}
	}
	if !foundPassport {
		t.Fatalf("Expected detected text to include passport keywords, got %v", out.Results)
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

	out, err := engine.DetectText("../../models/test_images/example1.png").
		WithLayoutReading(true).
		AsLines().
		Run()
	if err != nil {
		t.Fatalf("expected text detection to succeed even without layout model, got: %v", err)
	}
	if len(out.Results) == 0 {
		t.Fatal("expected text detection results, got 0")
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

	out, err := engine.DetectText(imgPath).WithLayoutReading(true).AsLines().Run()
	if err != nil {
		t.Fatalf("DetectText with layout failed: %v", err)
	}

	if len(out.Results) != 243 {
		t.Fatalf("Expected exactly 243 lines, got %d", len(out.Results))
	}

	if out.Results[0].Text != "All Orders in our" {
		t.Fatalf("Expected first line to be 'All Orders in our', got '%s'", out.Results[0].Text)
	}
	if out.Results[0].RegionID < 0 {
		t.Fatalf("Expected RegionID >= 0 when layout is enabled, got %d", out.Results[0].RegionID)
	}
	if out.Results[0].LayoutType < 0 {
		t.Fatalf("Expected LayoutType >= 0 when layout is enabled, got %d", out.Results[0].LayoutType)
	}
}

func TestDetectTextFluentBuilder(t *testing.T) {
	detPath := "../../models/PPOCR_v6_tiny/det.rten"
	recPath := "../../models/PPOCR_v6_tiny/rec.rten"
	dictPath := "../../models/PPOCR_v6_tiny/dict.txt"
	layoutPath := "../../models/PPOCR_v6_tiny/layout_cdla.onnx"
	imgPath := "../../tests/fixtures/chronam-ad-page.png"

	if _, err := os.Stat(detPath); os.IsNotExist(err) {
		t.Skip("Model files not found, skipping fluent builder test")
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

	descreen := float32(0.1)
	batchSize := int32(6)
	workers := uint32(2)
	maxWorkers := uint32(4)
	scoreThresh := float32(0.5)

	res, err := engine.DetectText().
		Image(imgPath).
		WithLayoutReading(true).
		WithPreDetection(PreDetectionOptions{DescreenStrength: &descreen}).
		WithRecognition(RecognitionRunOptions{BatchSize: &batchSize, Workers: &workers, MaxWorkers: &maxWorkers}).
		WithPostRecognition(PostRecognitionOptions{ScoreThreshold: &scoreThresh}).
		AsLines().
		Run()

	if err != nil {
		t.Fatalf("Fluent builder Run failed: %v", err)
	}

	if res.IsSpatial || len(res.Results) == 0 {
		t.Fatalf("Expected non-empty structured results from fluent builder")
	}
	if !strings.HasPrefix(res.Results[0].Text, "All Orders in") {
		t.Fatalf("Expected text to start with 'All Orders in', got: %s", res.Results[0].Text)
	}

	spatialRes, err := engine.DetectText().
		Image(imgPath).
		WithLayoutReading(true).
		AsSpatial().
		Run()

	if err != nil {
		t.Fatalf("Fluent builder AsSpatial().Run() failed: %v", err)
	}
	if !spatialRes.IsSpatial || !strings.Contains(spatialRes.Text, "Orders") {
		t.Fatalf("Expected spatial text to contain 'Orders', got: %s", spatialRes.Text)
	}
}

func TestVersionAndIsReady(t *testing.T) {
	v := Version()
	if v != "0.4.0" {
		t.Fatalf("Expected version '0.4.0', got: %s", v)
	}

	e := &Engine{handle: nil}
	if e.IsReady() {
		t.Fatal("Expected closed engine IsReady() to be false")
	}
	if e.Version() != "0.4.0" {
		t.Fatalf("Expected e.Version() '0.4.0', got: %s", e.Version())
	}
}

