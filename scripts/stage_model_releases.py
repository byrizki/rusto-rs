#!/usr/bin/env python3
"""
Stages OCR model assets into:
1. Hugging Face organized directory structure (model-releases-hf/)
2. GitHub Release flat directory structure (model-releases-flat/)
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Mapping of models, tiers, and file destinations
V6_TIERS = [
    ("ppocrv6-tiny", "PPOCR_v6_tiny", "ppocrv6_tiny", "PPOCR_v6_tiny/dict.txt"),
    ("ppocrv6-tiny-int8", "PPOCR_v6_tiny_int8", "ppocrv6_tiny_int8", "PPOCR_v6_tiny/dict.txt"),
    ("ppocrv6-small", "PPOCR_v6_small", "ppocrv6_small", "PPOCR_v6_small/dict.txt"),
    ("ppocrv6-small-int8", "PPOCR_v6_small_int8", "ppocrv6_small_int8", "PPOCR_v6_small/dict.txt"),
    ("ppocrv6-medium", "PPOCR_v6_medium", "ppocrv6_medium", "PPOCR_v6_medium/dict.txt"),
    ("ppocrv6-medium-int8", "PPOCR_v6_medium_int8", "ppocrv6_medium_int8", "PPOCR_v6_medium/dict.txt"),
]

V5_TIERS = [
    ("ppocrv5-mobile", "PPOCR_v5_mobile", "ppocrv5_mobile", "PPOCR_v5_mobile/dict.txt"),
    ("ppocrv5-server", "PPOCR_v5_server", "ppocrv5_server", "PPOCR_v5_mobile/dict.txt"),
]

V4_TIERS = [
    ("ppocrv4-mobile", "PPOCR_v4_mobile", "ppocrv4_mobile", "PPOCR_v4_mobile/dict.txt"),
    ("ppocrv4-server", "PPOCR_v4_server", "ppocrv4_server", "PPOCR_v4_mobile/dict.txt"),
]

V5_LANGS = ["arabic", "cyrillic", "devanagari", "el", "eslav", "korean", "latin", "ta", "te", "th"]
V4_LANGS = ["japan", "chinese_cht", "kannada"]

MODULE_EXTRAS = [
    ("cls.rten", "modules/cls.rten", "ppocr_cls.rten"),
    ("orient.rten", "modules/orient.rten", "ppocr_orient.rten"),
    ("uvdoc.rten", "modules/uvdoc.rten", "uvdoc.rten"),
    ("layout_cdla.rten", "modules/layout_cdla.rten", "layout_cdla.rten"),
    ("layout_publaynet.rten", "modules/layout_publaynet.rten", "layout_publaynet.rten"),
]

INDRA_ASSETS = [
    ("indra/det.rten", "det.rten", "indra_det.rten"),
    ("indra/rec.rten", "rec.rten", "indra_rec.rten"),
    ("indra/meta.json", "meta.json", "indra_meta.json"),
    ("indra/indra_order.rten", "order.rten", "indra_order.rten"),
    ("indra/orient.rten", "orient.rten", "indra_orient.rten"),
    ("indra/rec_cjk.rten", "rec_cjk.rten", "indra_rec_cjk.rten"),
    ("indra/rec_rtl.rten", "rec_rtl.rten", "indra_rec_rtl.rten"),
    ("indra/rec_world.rten", "rec_world.rten", "indra_rec_world.rten"),
    ("indra/script.rten", "script.rten", "indra_script.rten"),
    ("indra/mathcls.rten", "mathcls.rten", "indra_mathcls.rten"),
    ("indra/math_enc.rten", "math_enc.rten", "indra_math_enc.rten"),
    ("indra/math_dec_init.rten", "math_dec_init.rten", "indra_math_dec_init.rten"),
    ("indra/math_dec_step.rten", "math_dec_step.rten", "indra_math_dec_step.rten"),
    ("indra/math_meta.json", "math_meta.json", "indra_math_meta.json"),
]


def resolve_source(models_dir: Path, rel_path: str) -> Path | None:
    path = models_dir / rel_path
    if path.is_file() and path.stat().st_size > 0:
        return path
    # Check fallback in /tmp/hf_stage_rten_models if present
    tmp_path = Path("/tmp/hf_stage_rten_models") / Path(rel_path).name
    if tmp_path.is_file() and tmp_path.stat().st_size > 0:
        return tmp_path
    return None


def stage_asset(src: Path, hf_target: Path | None, flat_target: Path | None) -> None:
    if hf_target:
        hf_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, hf_target)
    if flat_target:
        flat_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, flat_target)


def stage_tier(models_dir: Path, hf_dir: Path, flat_dir: Path, tiers: list, v_dict: str) -> None:
    for folder, src_folder, prefix, default_dict in tiers:
        for kind in ["det", "rec"]:
            src = resolve_source(models_dir, f"{src_folder}/{kind}.rten")
            if not src:
                src = resolve_source(models_dir, f"{prefix}_{kind}.rten")
            if src:
                stage_asset(src, hf_dir / folder / f"{kind}.rten", flat_dir / f"{prefix}_{kind}.rten")
                stage_asset(src, hf_dir / folder / f"{prefix}_{kind}.rten", None)
            else:
                print(f"⚠️ Missing {folder}/{kind}.rten", file=sys.stderr)

        dict_src = resolve_source(models_dir, default_dict)
        if not dict_src:
            dict_src = resolve_source(models_dir, f"{v_dict}_dict.txt")
        if dict_src:
            stage_asset(dict_src, hf_dir / folder / "dict.txt", flat_dir / f"{v_dict}_dict.txt")
            stage_asset(dict_src, hf_dir / folder / f"{v_dict}_dict.txt", None)


def stage_multilingual(models_dir: Path, hf_dir: Path, flat_dir: Path) -> None:
    for lang in V5_LANGS:
        rec_src = resolve_source(models_dir, f"PPOCR_v5_{lang}/rec.rten") or resolve_source(models_dir, f"ppocrv5_rec_{lang}.rten")
        dict_src = resolve_source(models_dir, f"PPOCR_v5_{lang}/dict.txt") or resolve_source(models_dir, f"ppocrv5_dict_{lang}.txt")
        if rec_src:
            stage_asset(rec_src, hf_dir / "multilingual" / f"ppocrv5_rec_{lang}.rten", flat_dir / f"ppocrv5_rec_{lang}.rten")
        if dict_src:
            stage_asset(dict_src, hf_dir / "multilingual" / f"ppocrv5_dict_{lang}.txt", flat_dir / f"ppocrv5_dict_{lang}.txt")

    for lang in V4_LANGS:
        rec_src = resolve_source(models_dir, f"PPOCR_v4_{lang}/rec.rten") or resolve_source(models_dir, f"ppocrv4_rec_{lang}.rten")
        dict_src = resolve_source(models_dir, f"PPOCR_v4_{lang}/dict.txt") or resolve_source(models_dir, f"ppocrv4_dict_{lang}.txt")
        if rec_src:
            stage_asset(rec_src, hf_dir / "multilingual" / f"ppocrv4_rec_{lang}.rten", flat_dir / f"ppocrv4_rec_{lang}.rten")
        if dict_src:
            stage_asset(dict_src, hf_dir / "multilingual" / f"ppocrv4_dict_{lang}.txt", flat_dir / f"ppocrv4_dict_{lang}.txt")


def stage_extras(models_dir: Path, hf_dir: Path, flat_dir: Path) -> None:
    for rel_src, hf_dest, flat_name in MODULE_EXTRAS:
        src = resolve_source(models_dir, rel_src) or resolve_source(models_dir, flat_name)
        if src:
            stage_asset(src, hf_dir / hf_dest, flat_dir / flat_name)
            stage_asset(src, hf_dir / "modules" / flat_name, None)

    for rel_src, hf_dest, flat_name in INDRA_ASSETS:
        src = resolve_source(models_dir, rel_src) or resolve_source(models_dir, flat_name)
        if src:
            stage_asset(src, hf_dir / "indraocr" / hf_dest, flat_dir / flat_name)
            stage_asset(src, hf_dir / "indraocr" / flat_name, None)


def generate_sha256sums(directory: Path) -> None:
    sums = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS" and path.name != "README.md":
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            rel = path.relative_to(directory).as_posix()
            sums.append(f"{h.hexdigest()}  {rel}\n")
    with open(directory / "SHA256SUMS", "w") as f:
        f.writelines(sums)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage OCR models for Hugging Face and GitHub Release")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--hf-dir", type=Path, default=Path("model-releases-hf"))
    parser.add_argument("--flat-dir", type=Path, default=Path("model-releases-flat"))
    parser.add_argument("--readme", type=Path, default=Path("scripts/huggingface-models-readme.md"))
    args = parser.parse_args()

    args.hf_dir.mkdir(parents=True, exist_ok=True)
    args.flat_dir.mkdir(parents=True, exist_ok=True)

    print("=== Staging PP-OCR Base Tiers ===")
    stage_tier(args.models_dir, args.hf_dir, args.flat_dir, V6_TIERS, "ppocrv6")
    stage_tier(args.models_dir, args.hf_dir, args.flat_dir, V5_TIERS, "ppocrv5")
    stage_tier(args.models_dir, args.hf_dir, args.flat_dir, V4_TIERS, "ppocrv4")

    print("=== Staging Multilingual Models ===")
    stage_multilingual(args.models_dir, args.hf_dir, args.flat_dir)

    print("=== Staging Extras & Indra Models ===")
    stage_extras(args.models_dir, args.hf_dir, args.flat_dir)

    if args.readme.is_file():
        shutil.copyfile(args.readme, args.hf_dir / "README.md")

    print("=== Generating Checksums ===")
    generate_sha256sums(args.hf_dir)
    generate_sha256sums(args.flat_dir)
    print("✓ Model staging complete.")


if __name__ == "__main__":
    main()
