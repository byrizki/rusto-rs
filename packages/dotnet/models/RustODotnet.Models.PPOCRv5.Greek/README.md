# RustODotnet.Models.PPOCRv5.Greek

Pre-converted PP-OCRv5 Greek recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.onnx` (PP-OCRv5 Greek Recognition)
- `dict.txt` (Greek dictionary)

> **Note**: Detection model (`det.onnx`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv5.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
