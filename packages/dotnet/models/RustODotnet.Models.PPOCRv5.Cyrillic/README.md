# RustODotnet.Models.PPOCRv5.Cyrillic

Pre-converted PP-OCRv5 Cyrillic script (Russian etc.) MNN recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.mnn` (PP-OCRv5 Cyrillic Recognition)
- `dict.txt` (Cyrillic script (Russian etc.) dictionary)

> **Note**: Detection model (`det.mnn`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv5.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
