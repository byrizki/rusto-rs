import subprocess
import time
import statistics
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

BIN_NN = "./target/release/rusto"
BIN_RTEN = "./target/release/rusto-rten"

CHRONAM_IMG = "tests/fixtures/chronam-ad-page.png"
USATODAY_IMG = "benchmarks/omnidocbench/dataset/images/newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

# Configurations for Latency
LATENCY_CONFIGS = [
    {
        "name": "Chronam: PP-OCRv6 Tiny (Plain)",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: PP-OCRv6 Tiny (CDLA Layout)",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "--enable-layout",
            "--layout-model", "models/PPOCR_v6_tiny/layout_cdla.mnn",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "--enable-layout",
            "--layout-model", "models/PPOCR_v6_tiny/layout_cdla.onnx",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: Indra OCR (Plain)",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: Indra OCR (Neural Order)",
        "nn": [
            BIN_NN, CHRONAM_IMG,
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.mnn",
            "--reading-order",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, CHRONAM_IMG,
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.rten",
            "--reading-order",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: PP-OCRv6 Tiny (Plain)",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: PP-OCRv6 Tiny (CDLA Layout)",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "--enable-layout",
            "--layout-model", "models/PPOCR_v6_tiny/layout_cdla.mnn",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.onnx",
            "--rec-model", "models/PPOCR_v6_tiny/rec.onnx",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "--enable-layout",
            "--layout-model", "models/PPOCR_v6_tiny/layout_cdla.onnx",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: Indra OCR (Plain)",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: Indra OCR (Neural Order)",
        "nn": [
            BIN_NN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.mnn",
            "--reading-order",
            "-f", "text"
        ],
        "rten": [
            BIN_RTEN, USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.rten",
            "--rec-model", "models/indra/rec.rten",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.rten",
            "--reading-order",
            "-f", "text"
        ],
    },
]

def run_cmd(args):
    t0 = time.perf_counter()
    res = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000.0

def measure_latency(args, rounds=4):
    # Warmup
    run_cmd(args)
    times = [run_cmd(args) for _ in range(rounds)]
    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }

print("="*80)
print("PART 1: LATENCY BENCHMARK (Single-Call)")
print("="*80)
latency_results = []
for cfg in LATENCY_CONFIGS:
    print(f"Testing {cfg['name']}...")
    nn_m = measure_latency(cfg["nn"])
    rten_m = measure_latency(cfg["rten"])
    diff = nn_m["median"] - rten_m["median"]
    pct = (diff / rten_m["median"]) * 100.0
    status = f"rusto-nn is {-diff:.1f} ms FASTER ({abs(pct):.1f}%)" if diff < 0 else f"rusto-rten is {diff:.1f} ms faster ({pct:.1f}%)"
    print(f"  rusto-nn: {nn_m['median']:.1f} ms | rusto-rten: {rten_m['median']:.1f} ms -> {status}")
    latency_results.append({
        "name": cfg["name"],
        "nn": nn_m,
        "rten": rten_m,
        "diff": diff,
    })

print("\n" + "="*80)
print("PART 2: CONCURRENT THROUGHPUT BENCHMARK")
print("Target: Chronam Ad Page (PP-OCRv6 Tiny, 153 crops)")
print("="*80)

def run_concurrent_test(args, concurrency, total_requests):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_cmd, args) for _ in range(total_requests)]
        times = [f.result() for f in futures]
    wall_clock = time.perf_counter() - t0
    qps = total_requests / wall_clock
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_clock_s": wall_clock,
        "qps": qps,
        "mean_latency_ms": statistics.mean(times),
        "median_latency_ms": statistics.median(times),
    }

CONCURRENCIES = [1, 2, 4, 8]
TOTAL_REQUESTS = 8

nn_args_bench = LATENCY_CONFIGS[0]["nn"]
rten_args_bench = LATENCY_CONFIGS[0]["rten"]

concurrent_results = []
for c in CONCURRENCIES:
    print(f"Testing Concurrency = {c} ({TOTAL_REQUESTS} total requests)...")
    nn_c = run_concurrent_test(nn_args_bench, c, TOTAL_REQUESTS)
    rten_c = run_concurrent_test(rten_args_bench, c, TOTAL_REQUESTS)
    print(f"  rusto-nn:   QPS = {nn_c['qps']:.2f} req/s | Mean Latency = {nn_c['mean_latency_ms']:.1f} ms")
    print(f"  rusto-rten: QPS = {rten_c['qps']:.2f} req/s | Mean Latency = {rten_c['mean_latency_ms']:.1f} ms")
    concurrent_results.append({
        "concurrency": c,
        "nn": nn_c,
        "rten": rten_c,
    })

output_data = {
    "latency": latency_results,
    "concurrency": concurrent_results,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
results_file = os.path.join(RESULTS_DIR, "benchmark_results.json")

with open(results_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved results to {results_file}")

