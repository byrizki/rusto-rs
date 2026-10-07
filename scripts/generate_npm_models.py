#!/usr/bin/env python3
"""
Generate and synchronize npm packages for RustO pre-trained OCR models.
Mirrors model packages in Android, iOS, and .NET.

Usage:
  python3 scripts/generate_npm_models.py --core           # Generate 4 core presets
  python3 scripts/generate_npm_models.py --all            # Generate all 20 model packages
  python3 scripts/generate_npm_models.py --model ppocrv6-tiny
  python3 scripts/generate_npm_models.py --core --copy-models
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_MODELS_DIR = REPO_ROOT / "packages" / "models"
MODELS_SOURCE_DIR = REPO_ROOT / "models"

# 20 pre-trained model specifications
MODEL_DEFINITIONS = [
    {
        "id": "ppocrv6-tiny",
        "name": "rusto-models-ppocrv6-tiny",
        "title": "PP-OCRv6 Tiny",
        "description": "Pre-trained PP-OCRv6 Tiny models (~6 MB) for RustO OCR",
        "preset": "ppv6",
        "tier": "tiny",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v6_tiny", "PPOCR_v6"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": True,
    },
    {
        "id": "ppocrv6-small",
        "name": "rusto-models-ppocrv6-small",
        "title": "PP-OCRv6 Small",
        "description": "Pre-trained PP-OCRv6 Small models (~12 MB) for RustO OCR",
        "preset": "ppv6",
        "tier": "small",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v6_small"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": True,
    },
    {
        "id": "ppocrv6-medium",
        "name": "rusto-models-ppocrv6-medium",
        "title": "PP-OCRv6 Medium",
        "description": "Pre-trained PP-OCRv6 Medium models (~24 MB) for RustO OCR",
        "preset": "ppv6",
        "tier": "medium",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v6_medium"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": True,
    },
    {
        "id": "ppocrv5-mobile",
        "name": "rusto-models-ppocrv5-mobile",
        "title": "PP-OCRv5 Mobile",
        "description": "Pre-trained PP-OCRv5 Mobile models (~8 MB) for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v5_mobile", "PPOCR_v5"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "optional_files": ["rec_en.onnx", "dict_en.txt"],
        "is_core": True,
    },
    {
        "id": "ppocrv5-server",
        "name": "rusto-models-ppocrv5-server",
        "title": "PP-OCRv5 Server",
        "description": "Pre-trained PP-OCRv5 Server high-accuracy models for RustO OCR",
        "preset": "ppv5",
        "tier": "server",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v5_server"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-arabic",
        "name": "rusto-models-ppocrv5-arabic",
        "title": "PP-OCRv5 Arabic",
        "description": "Pre-trained PP-OCRv5 Arabic recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "arabic",
        "source_dirs": ["PPOCR_v5_arabic"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-cyrillic",
        "name": "rusto-models-ppocrv5-cyrillic",
        "title": "PP-OCRv5 Cyrillic",
        "description": "Pre-trained PP-OCRv5 Cyrillic recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "cyrillic",
        "source_dirs": ["PPOCR_v5_cyrillic"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-devanagari",
        "name": "rusto-models-ppocrv5-devanagari",
        "title": "PP-OCRv5 Devanagari",
        "description": "Pre-trained PP-OCRv5 Devanagari (Hindi) recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "devanagari",
        "source_dirs": ["PPOCR_v5_devanagari"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-el",
        "name": "rusto-models-ppocrv5-el",
        "title": "PP-OCRv5 Greek",
        "description": "Pre-trained PP-OCRv5 Greek recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "greek",
        "source_dirs": ["PPOCR_v5_el"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-eslav",
        "name": "rusto-models-ppocrv5-eslav",
        "title": "PP-OCRv5 East Slavic",
        "description": "Pre-trained PP-OCRv5 East Slavic recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "east_slavic",
        "source_dirs": ["PPOCR_v5_eslav"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-korean",
        "name": "rusto-models-ppocrv5-korean",
        "title": "PP-OCRv5 Korean",
        "description": "Pre-trained PP-OCRv5 Korean recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "korean",
        "source_dirs": ["PPOCR_v5_korean"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-latin",
        "name": "rusto-models-ppocrv5-latin",
        "title": "PP-OCRv5 Latin",
        "description": "Pre-trained PP-OCRv5 Latin multilingual recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "latin",
        "source_dirs": ["PPOCR_v5_latin"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-ta",
        "name": "rusto-models-ppocrv5-ta",
        "title": "PP-OCRv5 Tamil",
        "description": "Pre-trained PP-OCRv5 Tamil recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "tamil",
        "source_dirs": ["PPOCR_v5_ta"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-te",
        "name": "rusto-models-ppocrv5-te",
        "title": "PP-OCRv5 Telugu",
        "description": "Pre-trained PP-OCRv5 Telugu recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "telugu",
        "source_dirs": ["PPOCR_v5_te"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv5-th",
        "name": "rusto-models-ppocrv5-th",
        "title": "PP-OCRv5 Thai",
        "description": "Pre-trained PP-OCRv5 Thai recognition MNN model for RustO OCR",
        "preset": "ppv5",
        "tier": "mobile",
        "language": "thai",
        "source_dirs": ["PPOCR_v5_th"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv4-mobile",
        "name": "rusto-models-ppocrv4-mobile",
        "title": "PP-OCRv4 Mobile",
        "description": "Pre-trained PP-OCRv4 Mobile models for RustO OCR",
        "preset": "ppv4",
        "tier": "mobile",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v4_mobile"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv4-server",
        "name": "rusto-models-ppocrv4-server",
        "title": "PP-OCRv4 Server",
        "description": "Pre-trained PP-OCRv4 Server high-accuracy models for RustO OCR",
        "preset": "ppv4",
        "tier": "server",
        "language": "multilingual",
        "source_dirs": ["PPOCR_v4_server"],
        "files": ["det.onnx", "rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv4-japan",
        "name": "rusto-models-ppocrv4-japan",
        "title": "PP-OCRv4 Japanese",
        "description": "Pre-trained PP-OCRv4 Japanese recognition MNN model for RustO OCR",
        "preset": "ppv4",
        "tier": "mobile",
        "language": "japanese",
        "source_dirs": ["PPOCR_v4_japan"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv4-chinese-cht",
        "name": "rusto-models-ppocrv4-chinese-cht",
        "title": "PP-OCRv4 Traditional Chinese",
        "description": "Pre-trained PP-OCRv4 Traditional Chinese recognition MNN model for RustO OCR",
        "preset": "ppv4",
        "tier": "mobile",
        "language": "chinese_traditional",
        "source_dirs": ["PPOCR_v4_chinese_cht"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
    {
        "id": "ppocrv4-kannada",
        "name": "rusto-models-ppocrv4-kannada",
        "title": "PP-OCRv4 Kannada",
        "description": "Pre-trained PP-OCRv4 Kannada recognition MNN model for RustO OCR",
        "preset": "ppv4",
        "tier": "mobile",
        "language": "kannada",
        "source_dirs": ["PPOCR_v4_kannada"],
        "files": ["rec.onnx", "dict.txt"],
        "is_core": False,
    },
]

def get_current_version() -> str:
    cargo_toml = REPO_ROOT / "Cargo.toml"
    with open(cargo_toml, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith('version = "'):
                return line.strip().split('"')[1]
    return "0.2.5"

def generate_package_json(definition: dict, version: str) -> dict:
    pkg_name = definition["name"]
    return {
        "name": pkg_name,
        "version": version,
        "description": definition["description"],
        "keywords": [
            "ocr",
            "rusto",
            "models",
            "paddleocr",
            "onnx", "rten",
            definition["preset"],
            definition["tier"],
            definition["language"],
        ],
        "homepage": "https://github.com/byrizki/rusto-rs#readme",
        "bugs": {
            "url": "https://github.com/byrizki/rusto-rs/issues"
        },
        "license": "MIT",
        "author": "byrizki <contact@byrizki.com> (https://github.com/byrizki)",
        "repository": {
            "type": "git",
            "url": "git+https://github.com/byrizki/rusto-rs.git",
            "directory": f"packages/models/{pkg_name}"
        },
        "main": "dist/index.js",
        "module": "dist/index.mjs",
        "types": "dist/index.d.ts",
        "exports": {
            ".": {
                "types": "./dist/index.d.ts",
                "import": "./dist/index.mjs",
                "require": "./dist/index.js"
            },
            "./models/*": "./models/*",
            "./package.json": "./package.json"
        },
        "files": [
            "dist",
            "models",
            "README.md",
            "LICENSE"
        ],
        "publishConfig": {
            "access": "public",
            "registry": "https://registry.npmjs.org/"
        },
        "scripts": {
            "build": "tsc -p tsconfig.build.json",
            "typecheck": "tsc --noEmit",
            "clean": "rm -rf dist"
        },
        "devDependencies": {
            "typescript": "^5.0.2"
        }
    }

def generate_tsconfig_json() -> dict:
    return {
        "compilerOptions": {
            "target": "ES2020",
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "lib": ["ES2020", "DOM"],
            "declaration": True,
            "declarationMap": True,
            "sourceMap": True,
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "rootDir": "src",
            "outDir": "dist"
        },
        "include": ["src/**/*"]
    }

def generate_tsconfig_build_json() -> dict:
    return {
        "extends": "./tsconfig.json",
        "compilerOptions": {
            "noEmit": False
        },
        "exclude": ["**/*.test.ts"]
    }

def generate_src_index_ts(definition: dict, version: str) -> str:
    files_dict = {}
    for f in definition["files"]:
        if f.startswith("det."):
            files_dict["detection"] = f"models/{f}"
        elif f.startswith("rec."):
            files_dict["recognition"] = f"models/{f}"
        elif f.startswith("dict.") or f.endswith("_dict.txt") or f == "dict.txt":
            files_dict["dictionary"] = f"models/{f}"
        elif f.startswith("cls."):
            files_dict["classification"] = f"models/{f}"

    for opt_f in definition.get("optional_files", []):
        if opt_f.startswith("rec_en."):
            files_dict["recognitionEnglish"] = f"models/{opt_f}"
        elif opt_f.startswith("dict_en."):
            files_dict["dictionaryEnglish"] = f"models/{opt_f}"

    files_json = json.dumps(files_dict, indent=4)
    pkg_name = definition["name"]
    title = definition["title"]
    preset = definition["preset"]
    tier = definition["tier"]
    language = definition["language"]

    return f"""/**
 * RustO Pre-trained Model Package: {title}
 * Package: {pkg_name}
 */

export interface ModelFilesConfig {{
  detection?: string;
  recognition?: string;
  dictionary?: string;
  classification?: string;
  recognitionEnglish?: string;
  dictionaryEnglish?: string;
  [key: string]: string | undefined;
}}

export interface ModelPackageMetadata {{
  name: string;
  version: string;
  title: string;
  preset: string;
  tier: string;
  language: string;
  files: ModelFilesConfig;
}}

export const modelMetadata: ModelPackageMetadata = {{
  name: '{pkg_name}',
  version: '{version}',
  title: '{title}',
  preset: '{preset}',
  tier: '{tier}',
  language: '{language}',
  files: {files_json},
}};

/**
 * Returns package-relative asset paths for the bundled model files.
 */
export function getModelRelativePaths(): ModelFilesConfig {{
  return {{ ...modelMetadata.files }};
}}

/**
 * Resolve absolute URLs given a host base URL for browser/CDN environments.
 */
export function getModelUrls(baseUrl: string = ''): ModelFilesConfig {{
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const result: ModelFilesConfig = {{}};
  for (const [key, relativePath] of Object.entries(modelMetadata.files)) {{
    if (relativePath) {{
      result[key] = normalizedBase ? `${{normalizedBase}}/${{relativePath}}` : relativePath;
    }}
  }}
  return result;
}}

/**
 * Fetch and load binary buffers in browser or Node environments.
 */
export async function loadModelBuffers(baseUrlOrPath: string = ''): Promise<Record<string, Uint8Array | string>> {{
  const urls = getModelUrls(baseUrlOrPath);
  const buffers: Record<string, Uint8Array | string> = {{}};

  for (const [key, target] of Object.entries(urls)) {{
    if (!target) continue;

    if (typeof window !== 'undefined' || typeof fetch === 'function') {{
      const res = await fetch(target);
      if (!res.ok) {{
        throw new Error(`Failed to fetch model asset [${{key}}] from ${{target}}: ${{res.statusText}}`);
      }}
      if (key.includes('dict') || target.endsWith('.txt')) {{
        buffers[key] = await res.text();
      }} else {{
        const ab = await res.arrayBuffer();
        buffers[key] = new Uint8Array(ab);
      }}
    }}
  }}

  return buffers;
}}

export default modelMetadata;
"""

def generate_readme_md(definition: dict, version: str) -> str:
    pkg_name = definition["name"]
    title = definition["title"]
    preset = definition["preset"]
    tier = definition["tier"]
    lang = definition["language"]

    return f"""# {pkg_name}

> Pre-trained {title} model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `{preset}`
- **Model Tier:** `{tier}`
- **Target Language:** `{lang}`
- **Version:** `{version}`

## Installation

```bash
npm install {pkg_name}
# or
yarn add {pkg_name}
# or
pnpm add {pkg_name}
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import {{ modelMetadata, getModelUrls }} from '{pkg_name}';
import {{ initialize }} from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/{pkg_name}');

await initialize({{
  preset: '{preset}',
  models: {{
    detection: urls.detection,
    recognition: urls.recognition,
    dictionary: urls.dictionary,
  }},
}});
```

### In Node.js / Server environments

```ts
import {{ modelMetadata, getModelRelativePaths }} from '{pkg_name}';

console.log(modelMetadata.files);
```

## License

MIT License
"""

def generate_model_package(definition: dict, version: str, copy_models: bool = False):
    pkg_dir = PACKAGES_MODELS_DIR / definition["name"]
    src_dir = pkg_dir / "src"
    models_dir = pkg_dir / "models"
    
    src_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. package.json
    pkg_json = generate_package_json(definition, version)
    with open(pkg_dir / "package.json", "w", encoding="utf-8") as f:
        json.dump(pkg_json, f, indent=2)
        f.write("\n")

    # 2. tsconfig.json & tsconfig.build.json
    with open(pkg_dir / "tsconfig.json", "w", encoding="utf-8") as f:
        json.dump(generate_tsconfig_json(), f, indent=2)
        f.write("\n")

    with open(pkg_dir / "tsconfig.build.json", "w", encoding="utf-8") as f:
        json.dump(generate_tsconfig_build_json(), f, indent=2)
        f.write("\n")

    # 3. src/index.ts
    src_code = generate_src_index_ts(definition, version)
    with open(src_dir / "index.ts", "w", encoding="utf-8") as f:
        f.write(src_code)

    # 4. README.md
    readme_content = generate_readme_md(definition, version)
    with open(pkg_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 5. Copy license if available
    root_license = REPO_ROOT / "LICENSE"
    if root_license.exists():
        shutil.copy2(root_license, pkg_dir / "LICENSE")

    # 6. Copy models if requested and available
    if copy_models:
        copied_any = False
        for sdir_name in definition["source_dirs"]:
            source_candidate = MODELS_SOURCE_DIR / sdir_name
            if source_candidate.exists() and source_candidate.is_dir():
                for f_name in definition["files"] + definition.get("optional_files", []):
                    src_file = source_candidate / f_name
                    if src_file.exists():
                        shutil.copy2(src_file, models_dir / f_name)
                        copied_any = True
                if copied_any:
                    break

    print(f"  [OK] Generated npm package {definition['name']} -> {pkg_dir.relative_to(REPO_ROOT)}")

def main():
    parser = argparse.ArgumentParser(description="Generate npm packages for RustO pre-trained models")
    parser.add_argument("--all", action="store_true", help="Generate all 20 model packages")
    parser.add_argument("--core", action="store_true", help="Generate core model packages (ppocrv6-tiny, ppocrv6-small, ppocrv6-medium, ppocrv5-mobile)")
    parser.add_argument("--model", type=str, help="Generate a specific model package by id (e.g. ppocrv6-tiny)")
    parser.add_argument("--copy-models", action="store_true", help="Copy available model assets from repo models/ directory")
    args = parser.parse_args()

    version = get_current_version()
    print(f"RustO Model NPM Package Generator (Version: {version})")

    if not args.all and not args.core and not args.model:
        # Default to core
        args.core = True

    targets = []
    if args.model:
        targets = [d for d in MODEL_DEFINITIONS if d["id"] == args.model or d["name"] == args.model]
        if not targets:
            print(f"Error: Model '{args.model}' not found in definitions.", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        targets = MODEL_DEFINITIONS
    elif args.core:
        targets = [d for d in MODEL_DEFINITIONS if d["is_core"]]

    PACKAGES_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for definition in targets:
        generate_model_package(definition, version, copy_models=args.copy_models)

    print(f"Successfully generated {len(targets)} model package(s).")

if __name__ == "__main__":
    main()
