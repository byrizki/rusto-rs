# RustODotnet.Models.PPOCRv4.Mobile

Pre-converted PP-OCRv4 mobile MNN model files (including text orientation classifier) for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `det.mnn` (PP-OCRv4 Mobile Detection ~4.6 MB)
- `rec.mnn` (PP-OCRv4 Chinese+English Recognition ~10.5 MB)
- `rec_en.mnn` (PP-OCRv4 English Recognition ~7.3 MB)
- `cls.mnn` (PP-OCRv4 Direction / Orientation Classifier ~519 KB)
- `dict.txt` (Chinese dictionary)
- `dict_en.txt` (English dictionary)

When referenced, models are automatically copied to your output `models/` directory at build time.
