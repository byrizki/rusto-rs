# RustODotnet.Models.PPOCRv4.Server

Pre-converted PP-OCRv4 server high-accuracy MNN model files (including text orientation classifier) for [RustODotnet](https://www.nuget.org/packages/RustODotnet).

## Included Models

- `det.mnn` (PP-OCRv4 Server Detection ~112 MB)
- `rec.mnn` (PP-OCRv4 Chinese+English Server Recognition ~93 MB)
- `rec_en.mnn` (PP-OCRv4 English Server Recognition ~98 MB)
- `cls.mnn` (PP-OCRv4 Server Direction / Orientation Classifier ~1.6 MB)
- `dict.txt` (Chinese dictionary)
- `dict_en.txt` (English dictionary)

When referenced, models are automatically copied to your output `models/` directory at build time.
