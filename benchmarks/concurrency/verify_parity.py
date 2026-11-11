import subprocess
import json
import os
import difflib

BIN_NN = "./target/release/rusto"
BIN_RTEN = "./target/release/rusto-rten"
CHRONAM_IMG = "tests/fixtures/chronam-ad-page.png"
USATODAY_IMG = "benchmarks/omnidocbench/dataset/images/newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

CONFIGS = [
    {
        "name": "Chronam: PP-OCRv6 Tiny",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "json"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "json"
        ],
    },
    {
        "name": "Chronam: Indra OCR",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "json"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "-f", "json"
        ],
    },
    {
        "name": "USAToday 960: PP-OCRv6 Tiny",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "json"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "json"
        ],
    },
    {
        "name": "USAToday 960: Indra OCR",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "json"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "-f", "json"
        ],
    },
]

def run_json(cmd):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    return json.loads(res.stdout)

print("="*80)
print("EVALUATING OUTPUT TEXT PARITY (rusto-nn vs rusto-rten)")
print("="*80)

for cfg in CONFIGS:
    print(f"\n--- {cfg['name']} ---")
    data_nn = run_json(cfg["nn"])
    data_rten = run_json(cfg["rten"])

    texts_nn = [item["text"] for item in data_nn]
    texts_rten = [item["text"] for item in data_rten]

    full_nn = "\n".join(texts_nn)
    full_rten = "\n".join(texts_rten)

    matcher = difflib.SequenceMatcher(None, full_nn, full_rten)
    char_similarity = matcher.ratio() * 100.0

    exact_matches = sum(1 for a, b in zip(texts_nn, texts_rten) if a == b)
    total_lines = max(len(texts_nn), len(texts_rten))
    line_parity = (exact_matches / total_lines * 100.0) if total_lines > 0 else 100.0

    print(f"  Line count:      rusto-nn = {len(texts_nn)} | rusto-rten = {len(texts_rten)}")
    print(f"  Exact line match: {exact_matches}/{total_lines} ({line_parity:.1f}%)")
    print(f"  Character match: {char_similarity:.2f}%")

    if line_parity < 100.0:
        diff = list(difflib.unified_diff(texts_rten, texts_nn, fromfile="rten", tofile="nn", lineterm=""))
        print("  Sample differences (up to 6 lines):")
        for line in diff[:8]:
            print(f"    {line}")

