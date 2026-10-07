#!/usr/bin/env python3
import sys
import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

def decode_ctc(preds, character_list):
    # preds: [1, seq_len, num_classes]
    preds = preds[0]
    indices = np.argmax(preds, axis=-1)
    
    text = []
    prev_idx = None
    for idx in indices:
        if idx == prev_idx:
            continue
        prev_idx = idx
        if idx == 0 or idx >= len(character_list):
            continue
        text.append(character_list[idx])
    return "".join(text)

def main():
    root = Path(__file__).resolve().parent.parent.parent
    models_dir = root / "models" / "PPOCR_v6_tiny"
    det_model_path = str(models_dir / "det.onnx")
    rec_model_path = str(models_dir / "rec.onnx")
    dict_path = models_dir / "dict.txt"

    # Load character list
    character_list = ["blank"]
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            character_list.append(line.strip("\r\n"))
    character_list.append(" ")

    # Load tensors from binary files
    dump_dir = Path(__file__).resolve().parent / "dump"
    det_tensor = np.fromfile(dump_dir / "det_input.bin", dtype=np.float32).reshape(1, 3, 640, 640)
    
    with open(dump_dir / "meta.json", "r") as f:
        meta = json.load(f)
    num_crops = meta["num_crops"]

    crop_tensors = []
    for i in range(num_crops):
        crop = np.fromfile(dump_dir / f"crop_{i}.bin", dtype=np.float32).reshape(1, 3, 48, 320)
        crop_tensors.append(crop)

    # 1. Detection
    sess_det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
    inp_name_det = sess_det.get_inputs()[0].name
    
    # Warmup
    sess_det.run(None, {inp_name_det: det_tensor})
    
    # Benchmark 5 runs
    t0 = time.perf_counter()
    for _ in range(5):
        sess_det.run(None, {inp_name_det: det_tensor})
    det_latency_ms = (time.perf_counter() - t0) / 5.0 * 1000.0

    # 2. Recognition
    sess_rec = ort.InferenceSession(rec_model_path, providers=["CPUExecutionProvider"])
    inp_name_rec = sess_rec.get_inputs()[0].name
    
    # Warmup
    sess_rec.run(None, {inp_name_rec: crop_tensors[0]})

    rec_texts = []
    rec_latencies = []
    for i, crop in enumerate(crop_tensors):
        t0 = time.perf_counter()
        preds = sess_rec.run(None, {inp_name_rec: crop})[0]
        rec_latencies.append((time.perf_counter() - t0) * 1000.0)
        text = decode_ctc(preds, character_list)
        rec_texts.append(text)

    # Output json
    out = {
        "engine": "Direct ONNX (onnxruntime CPU)",
        "det_latency_ms": det_latency_ms,
        "avg_rec_latency_ms": float(np.mean(rec_latencies)),
        "total_rec_latency_ms": float(np.sum(rec_latencies)),
        "texts": rec_texts
    }
    with open(dump_dir / "onnx_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
