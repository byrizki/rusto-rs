import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

IMG = "tests/fixtures/chronam-ad-page.png"
NN = ["./target/release/rusto", IMG,
      "--det-model", "models/PPOCR_v6_tiny/det.mnn",
      "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
      "--dict", "models/PPOCR_v6_tiny/dict.txt", "-f", "text"]


def run(args, threads):
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if threads: env["RAYON_NUM_THREADS"] = str(threads)
    t0 = time.perf_counter()
    subprocess.run(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000


def sweep(concurrency, threads, total=8):
    run(NN, threads)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        lat = list(ex.map(lambda _: run(NN, threads), range(total)))
    wall = time.perf_counter() - t0
    return total / wall, statistics.mean(lat)


print("conc | rayon_threads | QPS | mean_latency_ms")
for conc in (1, 8):
    for threads in (0, 8):
        qps, lat = sweep(conc, threads)
        print(f"{conc:4d} | {threads:13d} | {qps:5.2f} | {lat:9.0f}", flush=True)

