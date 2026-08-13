# RustODotnet.Models.PPOCRv5.EastSlavic

Pre-converted PP-OCRv5 East Slavic MNN recognition model for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `rec.mnn` (PP-OCRv5 EastSlavic Recognition)
- `dict.txt` (East Slavic dictionary)

> **Note**: Detection model (`det.mnn`) is language-agnostic.
> Add a base model package (e.g. `RustODotnet.Models.PPOCRv5.Mobile`) for the detection model,
> then configure `recognition.modelPath` and `recognition.dictPath` to point to this package's files.

When referenced, models are automatically copied to your output `models/` directory at build time.
