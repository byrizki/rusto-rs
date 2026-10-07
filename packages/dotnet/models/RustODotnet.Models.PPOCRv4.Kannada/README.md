# RustODotnet.Models.PPOCRv4.Kannada

Pre-converted PP-OCRv4 Kannada script recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.onnx` (PP-OCRv4 Kannada Recognition)
- `dict.txt` (Kannada script dictionary)

> **Note**: Detection model (`det.onnx`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv4.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
