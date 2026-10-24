package rusto

import (
	"errors"
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
