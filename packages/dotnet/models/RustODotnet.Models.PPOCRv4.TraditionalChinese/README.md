# RustODotnet.Models.PPOCRv4.TraditionalChinese

Pre-converted PP-OCRv4 Traditional Chinese script recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.onnx` (PP-OCRv4 TraditionalChinese Recognition)
- `dict.txt` (Traditional Chinese script dictionary)

> **Note**: Detection model (`det.onnx`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv4.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
