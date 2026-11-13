# rusto-server

A small HTTP server that answers Mistral's `POST /v1/ocr` request shape with RustO (PP-OCRv6), so code written
against Mistral's OCR API can be pointed at a self-hosted server by changing the base URL. It is a separate
workspace crate on purpose: async and web dependencies stay out of `rusto-core` and the C/mobile outputs.

Origin: a prototype at [survos-sites/mistral-rusto](https://github.com/survos-sites/mistral-rusto), proposed in
#28 and measured in #29. Compatibility has only been tested with plain HTTP requests shaped like Mistral's, not
with Mistral's official SDKs.

## Supported

- `POST /v1/ocr` with `document.type` `image_url` or `document_url`: a `data:` URI (base64) or, for hosts listed in
  `OCR_ALLOWED_HOSTS`, an `http(s)` URL (no redirects, 32 MB cap). PNG and JPEG.
- Response: `pages[]` with `index`, `markdown` (the recognized lines in rusto's order), `images` (always empty) and
  `dimensions` (`dpi` is null), plus `model` and `usage_info`.
- Non-Mistral extras on each page, ignored by Mistral clients: `lines` (text, score, box), `blocks` (rusto's layout
  regions with their text; without a layout model every line is its own block), `orientation_corrected`, and
  `text_angle` (length-weighted median slope of the lines, in degrees; a skew measure, not a 90° detector).
- `model` chooses a model per request: a directory name (`tiny`, `small`), or `rusto-ppocrv6-<name>`. Mistral ids
  (`mistral-ocr-*`) and no model use the default; any other name is a 400. Models load on first use.
- `autorotate` (request field, non-Mistral): apply the EXIF orientation to the pixels before detection. Off by
  default, because EXIF tags are not always accurate; `OCR_AUTOROTATE=true` changes the server default.
- `GET /v1/models`, `GET /health`, optional bearer-token auth, Mistral's error envelope.

Not supported: PDFs, tables, headers and footers as separate output, image extraction, confidence options,
annotations, `pages`, and the `/v1/files` upload flow.

## Run

One subdirectory per model under a models directory, each with `det.onnx`, `rec.onnx` and `dict.txt`:

```bash
OCR_MODELS_DIR=/path/to/models cargo run -p rusto-server --release   # models/tiny, models/small
```

| Env | Default | |
|---|---|---|
| `OCR_MODELS_DIR` | | one subdirectory per model |
| `OCR_MODEL_DIR` | `models/tiny` | single-model shortcut, served as `tiny` (used when `OCR_MODELS_DIR` is unset) |
| `OCR_DEFAULT_MODEL` | `tiny` | |
| `OCR_LISTEN` | `127.0.0.1:8895` | |
| `OCR_POOL_SIZE` | `2` | engines per model; one engine is not safe to share across requests, so requests draw from a pool |
| `OCR_API_KEY` | unset | if set, `/v1/*` requires `Authorization: Bearer <key>` |
| `OCR_ALLOWED_HOSTS` | empty | hosts URLs may be fetched from; `data:` URIs always work |
| `OCR_AUTOROTATE` | `false` | server default for the `autorotate` request field |

## Test

```bash
OCR_MODELS_DIR=/path/to/models cargo test -p rusto-server --profile release-with-debug
```

The workspace `release` profile uses `panic = "abort"`, which the test harness cannot link against, and inference is
too slow unoptimized, so tests use `release-with-debug`. Tests that need models skip when `OCR_MODELS_DIR` is unset.
