import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

IMG = "tests/fixtures/chronam-ad-page.png"
ARGS = ["--det-model", "models/PPOCR_v6_tiny/det.mnn",
        "--rec-model", "models/PPOCR_v6_tiny/rec.mnn",
        "--dict", "models/PPOCR_v6_tiny/dict.txt", "-f", "text"]
BINS = {"rayon": "./target/release/rusto_rayon", "pool": "./target/release/rusto_pool"}


def timed(binary):
    t0 = time.perf_counter()
    subprocess.run([binary, IMG, *ARGS], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000


def single_stats(binary):
    out = subprocess.run(["/usr/bin/time", "-v", binary, IMG, *ARGS],
                         capture_output=True, text=True, check=True).stderr
    user = float(re.search(r"User time \(seconds\): ([\d.]+)", out).group(1))
    sys_t = float(re.search(r"System time \(seconds\): ([\d.]+)", out).group(1))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", out).group(1)) // 1024
    return user + sys_t, rss


def concurrent(binary, conc, total=8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        lat = list(ex.map(lambda _: timed(binary), range(total)))
    return total / (time.perf_counter() - t0), statistics.mean(lat)


print("variant | single_ms | cpu_s | rss_MB | C4_qps | C8_qps | C8_lat_ms", flush=True)
for name, binary in BINS.items():
    timed(binary)
    single = statistics.median(timed(binary) for _ in range(5))
    cpu, rss = single_stats(binary)
    q4, _ = concurrent(binary, 4)
    q8, l8 = concurrent(binary, 8)
    print(f"{name:7s} | {single:9.0f} | {cpu:5.2f} | {rss:6d} | {q4:6.2f} | {q8:6.2f} | {l8:9.0f}", flush=True)

