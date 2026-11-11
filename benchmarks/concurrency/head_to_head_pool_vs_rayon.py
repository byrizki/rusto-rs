import subprocess
import time
import statistics
import json
import os
import re
import difflib
from concurrent.futures import ThreadPoolExecutor

BIN_RAYON = "./target/release/rusto_rayon"
BIN_POOL = "./target/release/rusto_pool"

CHRONAM_IMG = "tests/fixtures/chronam-ad-page.png"
USATODAY_IMG = "benchmarks/omnidocbench/dataset/images/newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

CONFIGS = [
    {
        "name": "Chronam: PP-OCRv6 Tiny (Plain)",
        "args": [
            CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: PP-OCRv6 Tiny (CDLA Layout)",
        "args": [
            CHRONAM_IMG,
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "--enable-layout",
            "--layout-model", "models/PPOCR_v6_tiny/layout_cdla.mnn",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: Indra OCR (Plain)",
        "args": [
            CHRONAM_IMG,
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
    },
    {
        "name": "Chronam: Indra OCR (Neural Order)",
        "args": [
            CHRONAM_IMG,
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.mnn",
            "--reading-order",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: PP-OCRv6 Tiny (Plain)",
        "args": [
            USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/PPOCR_v6_tiny/det.mnn",
            "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
            "--dict", "models/PPOCR_v6_tiny/dict.txt",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: Indra OCR (Plain)",
        "args": [
            USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "-f", "text"
        ],
    },
    {
        "name": "USAToday 960: Indra OCR (Neural Order)",
        "args": [
            USATODAY_IMG,
            "--det-limit", "960",
            "--det-model", "models/indra/det.mnn",
            "--rec-model", "models/indra/rec.mnn",
            "--dict", "models/indra/meta.json",
            "--order-model", "models/indra/order.mnn",
            "--reading-order",
            "-f", "text"
        ],
    },
]

def run_cmd(binary, args):
    t0 = time.perf_counter()
    subprocess.run([binary] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000.0

def measure_latency(binary, args, rounds=4):
    run_cmd(binary, args) # Warmup
    times = [run_cmd(binary, args) for _ in range(rounds)]
    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }

def measure_resource(binary, args):
    cmd = ["/usr/bin/time", "-v", binary] + args
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    out = res.stderr
    user = float(re.search(r"User time \(seconds\): ([\d.]+)", out).group(1))
    sys_t = float(re.search(r"System time \(seconds\): ([\d.]+)", out).group(1))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", out).group(1)) // 1024
    faults = int(re.search(r"Minor \(reclaiming a frame\) page faults: (\d+)", out).group(1))
    return {
        "user_s": user,
        "sys_s": sys_t,
        "total_cpu_s": user + sys_t,
        "peak_rss_mb": rss,
        "minor_faults": faults,
    }

print("="*80)
print("PART 1: OUTPUT ACCURACY & PARITY (Rayon vs Mini Pool)")
print("="*80)

PARITY_CONFIGS = [
    ("Chronam PP-OCRv6 Tiny", CONFIGS[0]["args"]),
    ("Chronam Indra OCR", CONFIGS[2]["args"]),
    ("USAToday 960 Indra OCR", CONFIGS[5]["args"]),
]

for name, base_args in PARITY_CONFIGS:
    json_args = [a if a != "text" else "json" for a in base_args]
    out_r = subprocess.run([BIN_RAYON] + json_args, stdout=subprocess.PIPE, check=True, text=True).stdout
    out_p = subprocess.run([BIN_POOL] + json_args, stdout=subprocess.PIPE, check=True, text=True).stdout
    data_r = json.loads(out_r)
    data_p = json.loads(out_p)
    texts_r = [x["text"] for x in data_r]
    texts_p = [x["text"] for x in data_p]
    exact = sum(1 for a, b in zip(texts_r, texts_p) if a == b)
    total = max(len(texts_r), len(texts_p))
    char_ratio = difflib.SequenceMatcher(None, "\n".join(texts_r), "\n".join(texts_p)).ratio() * 100.0
    print(f"  {name:25s}: Lines = {total:3d} | Exact match = {exact:3d}/{total} ({exact/total*100.0:.1f}%) | Char match = {char_ratio:.2f}%")

print("\n" + "="*80)
print("PART 2: LATENCY & RESOURCE FOOTPRINT (Single-Call)")
print("="*80)

latency_results = []
for cfg in CONFIGS:
    r_lat = measure_latency(BIN_RAYON, cfg["args"])
    p_lat = measure_latency(BIN_POOL, cfg["args"])
    r_res = measure_resource(BIN_RAYON, cfg["args"])
    p_res = measure_resource(BIN_POOL, cfg["args"])
    diff = p_lat["median"] - r_lat["median"]
    pct = (diff / r_lat["median"]) * 100.0
    status = f"POOL is {-diff:.1f} ms FASTER ({abs(pct):.1f}%)" if diff < 0 else f"RAYON is {diff:.1f} ms faster ({pct:.1f}%)"
    print(f"{cfg['name']}:")
    print(f"  RAYON: {r_lat['median']:6.1f} ms | CPU: {r_res['total_cpu_s']:.2f}s | RSS: {r_res['peak_rss_mb']:3d} MB | Faults: {r_res['minor_faults']}")
    print(f"  POOL:  {p_lat['median']:6.1f} ms | CPU: {p_res['total_cpu_s']:.2f}s | RSS: {p_res['peak_rss_mb']:3d} MB | Faults: {p_res['minor_faults']}")
    print(f"  --> {status}")
    latency_results.append({
        "name": cfg["name"],
        "rayon_latency": r_lat,
        "pool_latency": p_lat,
        "rayon_res": r_res,
        "pool_res": p_res,
        "diff_ms": diff,
    })

print("\n" + "="*80)
print("PART 3: CONCURRENT THROUGHPUT (Chronam PP-OCRv6 Tiny, 8 Requests)")
print("="*80)

def run_concurrent(binary, args, concurrency, total=8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_cmd, binary, args) for _ in range(total)]
        times = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "qps": total / wall,
        "mean_latency_ms": statistics.mean(times),
    }

bench_args = CONFIGS[0]["args"]
concurrency_results = []
for c in [1, 2, 4, 8]:
    r_c = run_concurrent(BIN_RAYON, bench_args, c)
    p_c = run_concurrent(BIN_POOL, bench_args, c)
    print(f"Concurrency = {c}:")
    print(f"  RAYON: QPS = {r_c['qps']:.2f} req/s | Mean Latency = {r_c['mean_latency_ms']:.1f} ms")
    print(f"  POOL:  QPS = {p_c['qps']:.2f} req/s | Mean Latency = {p_c['mean_latency_ms']:.1f} ms")
    concurrency_results.append({
        "c": c,
        "rayon": r_c,
        "pool": p_c,
    })

results = {
    "latency": latency_results,
    "concurrency": concurrency_results,
}

out_path = "benchmarks/concurrency/results/head_to_head_pool_vs_rayon.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved detailed results to {out_path}")

