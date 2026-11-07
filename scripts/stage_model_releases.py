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

EXTENSIONS = [".mnn", ".onnx", ".rten"]

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

MODULE_SPECS = [
    ("cls", "ppocr_cls"),
    ("orient", "ppocr_orient"),
    ("uvdoc", "uvdoc"),
    ("layout_cdla", "layout_cdla"),
    ("layout_publaynet", "layout_publaynet"),
]

INDRA_SPECS = [
    ("det", "indra_det", ["det"]),
    ("rec", "indra_rec", ["rec"]),
    ("order", "indra_order", ["indra_order", "order"]),
    ("orient", "indra_orient", ["orient"]),
    ("rec_cjk", "indra_rec_cjk", ["rec_cjk"]),
    ("rec_rtl", "indra_rec_rtl", ["rec_rtl"]),
    ("rec_world", "indra_rec_world", ["rec_world"]),
    ("script", "indra_script", ["script"]),
    ("mathcls", "indra_mathcls", ["mathcls"]),
    ("math_enc", "indra_math_enc", ["math_enc"]),
    ("math_dec_init", "indra_math_dec_init", ["math_dec_init"]),
    ("math_dec_step", "indra_math_dec_step", ["math_dec_step"]),
]


def resolve_source(models_dir: Path, rel_path: str) -> Path | None:
    path = models_dir / rel_path
    if path.is_file() and path.stat().st_size > 0:
        return path
    tmp_path = Path("/tmp/hf_stage_rten_models") / Path(rel_path).name
    if tmp_path.is_file() and tmp_path.stat().st_size > 0:
        return tmp_path
    return None


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def stage_asset(src: Path, hf_target: Path | None, flat_target: Path | None) -> None:
    if hf_target:
        copy_or_link(src, hf_target)
    if flat_target:
        copy_or_link(src, flat_target)


def stage_tier(models_dir: Path, hf_dir: Path, flat_dir: Path, tiers: list, v_dict: str) -> None:
    for folder, src_folder, prefix, default_dict in tiers:
        for kind in ["det", "rec"]:
            found = False
            for ext in EXTENSIONS:
                src = (
                    resolve_source(models_dir, f"{src_folder}/{kind}{ext}")
                    or resolve_source(models_dir, f"{prefix}_{kind}{ext}")
                    or resolve_source(models_dir, f"{src_folder}/{prefix}_{kind}{ext}")
                )
                if src:
                    stage_asset(src, hf_dir / folder / f"{kind}{ext}", flat_dir / f"{prefix}_{kind}{ext}")
                    stage_asset(src, hf_dir / folder / f"{prefix}_{kind}{ext}", None)
                    found = True
            if not found:
                print(f"⚠️ Missing {folder}/{kind} across all extensions in {models_dir}", file=sys.stderr)

        dict_src = (
            resolve_source(models_dir, default_dict)
            or resolve_source(models_dir, f"{v_dict}_dict.txt")
            or resolve_source(models_dir, f"{src_folder}/dict.txt")
        )
        if dict_src:
            stage_asset(dict_src, hf_dir / folder / "dict.txt", flat_dir / f"{v_dict}_dict.txt")
            stage_asset(dict_src, hf_dir / folder / f"{v_dict}_dict.txt", None)


def stage_multilingual(models_dir: Path, hf_dir: Path, flat_dir: Path) -> None:
    for lang in V5_LANGS:
        for ext in EXTENSIONS:
            rec_src = (
                resolve_source(models_dir, f"PPOCR_v5_{lang}/rec{ext}")
                or resolve_source(models_dir, f"ppocrv5_rec_{lang}{ext}")
            )
            if rec_src:
                stage_asset(rec_src, hf_dir / "multilingual" / f"ppocrv5_rec_{lang}{ext}", flat_dir / f"ppocrv5_rec_{lang}{ext}")
        dict_src = (
            resolve_source(models_dir, f"PPOCR_v5_{lang}/dict.txt")
            or resolve_source(models_dir, f"ppocrv5_dict_{lang}.txt")
        )
        if dict_src:
            stage_asset(dict_src, hf_dir / "multilingual" / f"ppocrv5_dict_{lang}.txt", flat_dir / f"ppocrv5_dict_{lang}.txt")

    for lang in V4_LANGS:
        for ext in EXTENSIONS:
            rec_src = (
                resolve_source(models_dir, f"PPOCR_v4_{lang}/rec{ext}")
                or resolve_source(models_dir, f"ppocrv4_rec_{lang}{ext}")
            )
            if rec_src:
                stage_asset(rec_src, hf_dir / "multilingual" / f"ppocrv4_rec_{lang}{ext}", flat_dir / f"ppocrv4_rec_{lang}{ext}")
        dict_src = (
            resolve_source(models_dir, f"PPOCR_v4_{lang}/dict.txt")
            or resolve_source(models_dir, f"ppocrv4_dict_{lang}.txt")
        )
        if dict_src:
            stage_asset(dict_src, hf_dir / "multilingual" / f"ppocrv4_dict_{lang}.txt", flat_dir / f"ppocrv4_dict_{lang}.txt")


def stage_extras(models_dir: Path, hf_dir: Path, flat_dir: Path) -> None:
    for mod_name, flat_name in MODULE_SPECS:
        for ext in EXTENSIONS:
            src = (
                resolve_source(models_dir, f"{mod_name}{ext}")
                or resolve_source(models_dir, f"PPOCR_v6_tiny/{mod_name}{ext}")
                or resolve_source(models_dir, f"PPOCR_v4_mobile/{mod_name}{ext}")
                or resolve_source(models_dir, f"{flat_name}{ext}")
                or resolve_source(models_dir, f"modules/{mod_name}{ext}")
            )
            if src:
                stage_asset(src, hf_dir / "modules" / f"{mod_name}{ext}", flat_dir / f"{flat_name}{ext}")
                stage_asset(src, hf_dir / "modules" / f"{flat_name}{ext}", None)

    for canon_name, flat_name, candidate_prefixes in INDRA_SPECS:
        for ext in EXTENSIONS:
            src = None
            for pfx in candidate_prefixes:
                src = (
                    resolve_source(models_dir, f"indra/{pfx}{ext}")
                    or resolve_source(models_dir, f"{pfx}{ext}")
                    or resolve_source(models_dir, f"indraocr/{pfx}{ext}")
                )
                if src:
                    break
            if src:
                stage_asset(src, hf_dir / "indraocr" / f"{canon_name}{ext}", flat_dir / f"{flat_name}{ext}")
                stage_asset(src, hf_dir / "indraocr" / f"{flat_name}{ext}", None)

    for json_file, canon_dest, flat_dest in [
        ("indra/meta.json", "meta.json", "indra_meta.json"),
        ("indra/math_meta.json", "math_meta.json", "indra_math_meta.json"),
    ]:
        src = (
            resolve_source(models_dir, json_file)
            or resolve_source(models_dir, canon_dest)
            or resolve_source(models_dir, flat_dest)
        )
        if src:
            stage_asset(src, hf_dir / "indraocr" / canon_dest, flat_dir / flat_dest)
            stage_asset(src, hf_dir / "indraocr" / flat_dest, None)


def generate_sha256sums(directory: Path) -> None:
    sums = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS" and path.name != "README.md" and not path.name.startswith("."):
            # Avoid .cache/huggingface files
            if ".cache" in path.parts:
                continue
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
