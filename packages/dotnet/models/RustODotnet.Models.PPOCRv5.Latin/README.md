# RustODotnet.Models.PPOCRv5.Latin

Pre-converted PP-OCRv5 Latin script recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.onnx` (PP-OCRv5 Latin Recognition)
- `dict.txt` (Latin script dictionary)

> **Note**: Detection model (`det.onnx`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv5.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
