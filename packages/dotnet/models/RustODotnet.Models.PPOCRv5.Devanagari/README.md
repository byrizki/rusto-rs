# RustODotnet.Models.PPOCRv5.Devanagari

Pre-converted PP-OCRv5 Devanagari script (Hindi etc.) MNN recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.mnn` (PP-OCRv5 Devanagari Recognition)
- `dict.txt` (Devanagari script (Hindi etc.) dictionary)

> **Note**: Detection model (`det.mnn`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv5.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
