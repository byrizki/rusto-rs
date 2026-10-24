#!/usr/bin/env python3
"""
Packed Package Content Verifier for RustO Multi-Platform Distribution.
Validates the structural integrity, mandatory entrypoints, minimum non-zero sizes,
and absence of build cache/temporary leaks inside distribution archives:
- NPM packages (*.tgz)
- NuGet packages (*.nupkg)
- Android AARs (*.aar)
- iOS XCFramework Archives (*.zip)
- Python Wheels (*.whl)
"""

import argparse
import json
import os
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Common forbidden files that should never be present in release archives
COMMON_FORBIDDEN_PATTERNS = [
    r"\.DS_Store$",
    r"Thumbs\.db$",
    r"\.git/",
    r"\.gitignore$",
    r"__pycache__/",
    r"\.pytest_cache/",
    r"\.tsbuildinfo$",
    r"\.tmp$",
    r"\.orig$",
    r"\.swp$",
    r"/node_modules/",
    r"^node_modules/",
]

def check_forbidden_entries(names: List[str]) -> List[str]:
    violations = []
    for name in names:
        for pattern in COMMON_FORBIDDEN_PATTERNS:
            if re.search(pattern, name):
                violations.append(f"Forbidden entry: '{name}' matches pattern '{pattern}'")
    return violations

def verify_npm_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies an NPM tarball (.tgz)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            name_to_member = {m.name: m for m in members}

            # Check forbidden
            for f in check_forbidden_entries(names):
                errors.append(f)

            # npm packages nest under 'package/'
            pkg_json_name = next((n for n in names if n.endswith("package.json") and n.split("/")[0] == "package"), None)
            if not pkg_json_name:
                errors.append("Missing mandatory 'package/package.json'")
                return False, passed, errors

            # Read and parse package.json
            pkg_file = tar.extractfile(name_to_member[pkg_json_name])
            if not pkg_file:
                errors.append("Failed to extract package.json")
                return False, passed, errors

            try:
                pkg_data = json.loads(pkg_file.read().decode("utf-8"))
            except Exception as e:
                errors.append(f"Malformed package.json: {e}")
                return False, passed, errors

            name = pkg_data.get("name", "")
            version = pkg_data.get("version", "")
            passed.append(f"package.json valid: {name}@{version}")

            # Specific rules based on package name
            if "@rustors/react-native" in name or "react-native" in name:
                # Android build files
                has_android = any("android/build.gradle" in n for n in names)
                has_ios = any("ios/Rusto.mm" in n or "ios/Rusto.h" in n for n in names)
                has_rn_config = any("react-native.config.js" in n for n in names)
                has_podspec = any(n.endswith(".podspec") for n in names)

                if has_android:
                    passed.append("Contains Android bridge sources and build.gradle")
                else:
                    errors.append("Missing Android bridge files in React Native package")

                if has_ios:
                    passed.append("Contains iOS bridge sources (Rusto.mm/h)")
                else:
                    errors.append("Missing iOS bridge files in React Native package")

                if has_rn_config or has_podspec:
                    passed.append("Contains autolinking metadata (react-native.config.js / podspec)")
                else:
                    errors.append("Missing autolinking configuration")

            elif "@rustors/node" in name or "node" in name:
                has_index = any(re.search(r"package/index\.(c?js|ts)$", n) for n in names)
                has_types = any(re.search(r"package/index\.d\.(ts|cts)$", n) for n in names)
                if has_index:
                    passed.append("Contains Node entrypoint (index.js)")
                else:
                    errors.append("Missing Node entrypoint (index.js)")
                if has_types:
                    passed.append("Contains TypeScript declaration (index.d.ts)")
                else:
                    errors.append("Missing TypeScript declaration (index.d.ts)")

            elif "@rustors/web" in name or "@rustors/react" in name:
                has_dist = any("dist/" in n for n in names)
                if has_dist:
                    passed.append("Contains compiled dist/ bundle")
                else:
                    errors.append("Missing dist/ bundle in web/react package")

            elif "@rustors/model" in name or "model" in name:
                has_index = any(re.search(r"package/(dist/)?index\.(c?js|ts)$", n) for n in names)
                has_types = any(re.search(r"package/(dist/)?index\.d\.(ts|cts)$", n) for n in names)
                if has_index:
                    passed.append("Contains entrypoint (index.js / dist/index.js)")
                else:
                    errors.append("Missing entrypoint in model package")
                if has_types:
                    passed.append("Contains TypeScript declaration (index.d.ts / dist/index.d.ts)")
                else:
                    errors.append("Missing TypeScript declaration in model package")

                model_entries = [n for n in names if n.endswith(".rten") or n.endswith(".onnx") or n.endswith(".txt")]
                if model_entries:
                    passed.append(f"Contains {len(model_entries)} bundled model file(s): {', '.join(model_entries[:4])}")


            # Check that files have non-zero size
            for m in members:
                if m.isfile() and m.size == 0 and not m.name.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {m.name}")

    except Exception as e:
        errors.append(f"Failed to open/parse tarball: {e}")

    return len(errors) == 0, passed, errors

def verify_nuget_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a NuGet package (.nupkg)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            # Check nuspec
            nuspec_names = [n for n in names if n.endswith(".nuspec")]
            if not nuspec_names:
                errors.append("Missing .nuspec manifest")
            else:
                passed.append(f"Found nuspec: {nuspec_names[0]}")
                # Parse nuspec
                with z.open(nuspec_names[0]) as nuspec_file:
                    tree = ET.parse(nuspec_file)
                    root = tree.getroot()
                    # Check metadata id and version
                    metadata = root.find("{*}metadata")
                    if metadata is not None:
                        pkg_id = metadata.findtext("{*}id")
                        ver = metadata.findtext("{*}version")
                        passed.append(f"NuGet metadata: {pkg_id} v{ver}")

            # Check assemblies or runtimes
            dll_entries = [n for n in names if n.endswith(".dll")]
            if dll_entries:
                passed.append(f"Contains {len(dll_entries)} managed assembly/ies: {', '.join(dll_entries[:3])}")
            elif any("models" in archive_path.name.lower() for _ in [1]):
                model_entries = [n for n in names if n.endswith(".rten") or n.endswith(".onnx") or n.endswith(".txt")]
                if model_entries:
                    passed.append(f"Contains {len(model_entries)} model file(s): {', '.join(model_entries[:4])}")
                else:
                    errors.append("Model NuGet package contains no model files (.rten, .onnx, or .txt)")
            else:
                errors.append("Missing managed .dll assemblies in lib/")

            # Verify non-zero sizes
            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse nupkg: {e}")

    return len(errors) == 0, passed, errors

def verify_aar_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies an Android AAR archive."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            if "AndroidManifest.xml" in names:
                passed.append("Contains AndroidManifest.xml")
            else:
                errors.append("Missing AndroidManifest.xml")

            if "classes.jar" in names:
                passed.append("Contains classes.jar")
                # Verify classes.jar is not empty
                info = z.getinfo("classes.jar")
                if info.file_size < 1024:
                    errors.append(f"classes.jar suspiciously small ({info.file_size} bytes)")
            else:
                # Some model AARs might only contain assets
                if "model" not in archive_path.name.lower():
                    errors.append("Missing classes.jar in core Android AAR")

            # Check JNI libraries
            so_files = [n for n in names if n.endswith(".so")]
            if so_files:
                passed.append(f"Contains native libraries: {', '.join(so_files)}")
            elif "model" in archive_path.name.lower():
                asset_models = [n for n in names if ("assets/" in n) and (n.endswith(".rten") or n.endswith(".onnx") or n.endswith(".txt"))]
                if asset_models:
                    passed.append(f"Contains {len(asset_models)} model asset(s) in assets/: {', '.join(asset_models[:4])}")
                else:
                    errors.append("Model AAR contains no model assets (.rten, .onnx, or .txt) in assets/")
            else:
                errors.append("Core Android AAR contains no native .so libraries in jni/")

            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    # R.txt can legitimately be 0 bytes in pure-code libraries
                    if info.filename != "R.txt":
                        errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse AAR: {e}")

    return len(errors) == 0, passed, errors

def verify_xcframework_zip(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies an iOS XCFramework zip archive."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            plist = any(re.search(r"RustO\.xcframework/Info\.plist$", n) for n in names)
            if plist:
                passed.append("Contains RustO.xcframework/Info.plist")
            else:
                errors.append("Missing RustO.xcframework/Info.plist")

            libs = [n for n in names if n.endswith(".a") or n.endswith(".dylib")]
            if libs:
                passed.append(f"Contains XCFramework binary slices: {len(libs)} found")
            else:
                errors.append("Missing binary static libraries (.a) in XCFramework")

            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse XCFramework zip: {e}")

    return len(errors) == 0, passed, errors

def verify_model_zip(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a model resource bundle zip archive (e.g. iOS RustO-Models-*.zip)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            model_files = [n for n in names if n.endswith(".rten") or n.endswith(".onnx") or n.endswith(".txt")]
            if model_files:
                passed.append(f"Contains {len(model_files)} model/dictionary file(s): {', '.join(model_files[:4])}")
            else:
                errors.append("No .rten, .onnx, or .txt model files found in model zip")

            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse model zip: {e}")

    return len(errors) == 0, passed, errors

def verify_wheel_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a Python wheel (.whl)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            metadata_entry = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
            if metadata_entry:
                passed.append(f"Found wheel metadata: {metadata_entry}")
            else:
                errors.append("Missing .dist-info/METADATA in wheel")

            init_py = any(n.endswith("__init__.py") for n in names)
            if init_py:
                passed.append("Contains Python module initialization (__init__.py)")
            else:
                errors.append("Missing __init__.py in package wheel")

            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse Wheel: {e}")

    return len(errors) == 0, passed, errors

def verify_sdist_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a Python sdist package (.tar.gz)."""
    passed: List[str] = []
    errors: List[str] = []
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            for f in check_forbidden_entries(names):
                errors.append(f)
            if any(n.endswith("PKG-INFO") for n in names):
                passed.append("Contains PKG-INFO manifest")
            else:
                errors.append("Missing PKG-INFO in Python sdist")
            if any(n.endswith("pyproject.toml") or n.endswith("setup.py") for n in names):
                passed.append("Contains build manifest (pyproject.toml/setup.py)")
            else:
                errors.append("Missing build configuration in Python sdist")
            for m in members:
                if m.isfile() and m.size == 0 and not m.name.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {m.name}")
    except Exception as e:
        errors.append(f"Failed to open sdist archive: {e}")
    return len(errors) == 0, passed, errors

def verify_archive_bundle(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a raw binary bundle or wasm archive (.tar.gz)."""
    passed: List[str] = []
    errors: List[str] = []
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            for f in check_forbidden_entries(names):
                errors.append(f)
            if not names:
                errors.append("Empty archive bundle")
            else:
                passed.append(f"Contains {len(names)} bundle entry/ies")
            for m in members:
                if m.isfile() and m.size == 0 and not m.name.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {m.name}")
    except Exception as e:
        errors.append(f"Failed to open archive: {e}")
    return len(errors) == 0, passed, errors

def verify_jar_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a Java archive (.jar)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()

            for f in check_forbidden_entries(names):
                errors.append(f)

            if "META-INF/MANIFEST.MF" in names:
                passed.append("Contains META-INF/MANIFEST.MF")
            else:
                errors.append("Missing META-INF/MANIFEST.MF in JAR")

            class_files = [n for n in names if n.endswith(".class")]
            if class_files:
                passed.append(f"Contains {len(class_files)} compiled bytecode class(es): {', '.join(class_files[:3])}")
            else:
                errors.append("Missing compiled .class files in JAR")

            for info in z.infolist():
                if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {info.filename}")

    except Exception as e:
        errors.append(f"Failed to open/parse JAR: {e}")

    return len(errors) == 0, passed, errors

def parse_pubspec_fields(content: str) -> Dict[str, str]:
    fields = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k in ["name", "version", "description", "homepage"]:
                fields[k] = v
    return fields

def verify_pub_package(archive_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a Dart or Flutter pub package (.tar.gz / .tgz)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            name_to_member = {m.name: m for m in members}

            for f in check_forbidden_entries(names):
                errors.append(f)

            pubspec_name = next((n for n in names if n == "pubspec.yaml" or n.endswith("/pubspec.yaml")), None)
            if not pubspec_name:
                errors.append("Missing mandatory 'pubspec.yaml'")
                return False, passed, errors

            pubspec_file = tar.extractfile(name_to_member[pubspec_name])
            if not pubspec_file:
                errors.append("Failed to extract pubspec.yaml")
                return False, passed, errors

            content = pubspec_file.read().decode("utf-8")
            fields = parse_pubspec_fields(content)
            name = fields.get("name", "")
            version = fields.get("version", "")

            if not name:
                errors.append("pubspec.yaml missing 'name'")
            else:
                passed.append(f"pubspec.yaml valid: {name} v{version}")

            has_license = any(re.search(r"(^|/)LICENSE", n) for n in names)
            if has_license:
                passed.append("Contains LICENSE file")
            else:
                errors.append("Missing mandatory LICENSE file in pub package")

            has_readme = any(re.search(r"(^|/)README\.md", n) for n in names)
            if has_readme:
                passed.append("Contains README.md")
            else:
                errors.append("Missing README.md in pub package")

            has_lib = any(re.search(r"lib/.*\.dart$", n) for n in names)
            if has_lib:
                passed.append("Contains Dart library code in lib/")
            else:
                errors.append("Missing Dart source files in lib/")

            if "flutter:" in content or "rusto_flutter" in name:
                has_android = any(n.startswith("android/") or "/android/" in n for n in names)
                if has_android:
                    passed.append("Contains Flutter Android plugin bridge")

            for m in members:
                if m.isfile() and m.size == 0 and not m.name.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {m.name}")

    except Exception as e:
        errors.append(f"Failed to open/parse pub package archive: {e}")

    return len(errors) == 0, passed, errors

def verify_go_package(target_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Verifies a Go module directory or module archive (.zip)."""
    passed: List[str] = []
    errors: List[str] = []

    try:
        if target_path.is_file() and target_path.suffix == ".zip":
            with zipfile.ZipFile(target_path, "r") as z:
                names = z.namelist()
                for f in check_forbidden_entries(names):
                    errors.append(f)

                go_mod = next((n for n in names if n.endswith("go.mod")), None)
                if not go_mod:
                    errors.append("Missing go.mod in Go archive")
                else:
                    passed.append(f"Found go.mod: {go_mod}")

                go_files = [n for n in names if n.endswith(".go")]
                if go_files:
                    passed.append(f"Contains {len(go_files)} Go source file(s)")
                else:
                    errors.append("Missing Go source files (.go)")

                for info in z.infolist():
                    if not info.is_dir() and info.file_size == 0 and not info.filename.endswith(".gitkeep"):
                        errors.append(f"Empty 0-byte file found: {info.filename}")
        else:
            names = [str(p.relative_to(target_path)) for p in target_path.rglob("*")]
            for f in check_forbidden_entries(names):
                errors.append(f)

            go_mod_file = target_path / "go.mod"
            if not go_mod_file.is_file():
                errors.append("Missing go.mod in Go module directory")
            else:
                content = go_mod_file.read_text(encoding="utf-8")
                module_line = next((l for l in content.splitlines() if l.startswith("module ")), None)
                if module_line:
                    passed.append(f"Valid go.mod ({module_line.strip()})")
                else:
                    errors.append("go.mod missing 'module' directive")

            go_files = list(target_path.glob("*.go"))
            if go_files:
                passed.append(f"Contains {len(go_files)} Go source file(s)")
            else:
                errors.append("Missing Go source files (.go)")

            h_files = list(target_path.glob("*.h"))
            if h_files:
                passed.append(f"Contains C header(s): {', '.join(h.name for h in h_files)}")

            for p in target_path.rglob("*"):
                if p.is_file() and p.stat().st_size == 0 and not p.name.endswith(".gitkeep"):
                    errors.append(f"Empty 0-byte file found: {p.relative_to(target_path)}")

    except Exception as e:
        errors.append(f"Failed to verify Go package: {e}")

    return len(errors) == 0, passed, errors

def verify_file(file_path: Path) -> bool:
    name = file_path.name
    print(f"\n🔍 Verifying: {file_path}")

    if not file_path.exists() or (file_path.is_file() and file_path.stat().st_size == 0):
        print(f"❌ Error: Path does not exist or is 0 bytes: {file_path}")
        return False

    success = False
    passed: List[str] = []
    errors: List[str] = []

    if file_path.is_dir():
        if (file_path / "go.mod").exists():
            success, passed, errors = verify_go_package(file_path)
        else:
            return scan_directory(file_path)
    elif name.endswith(".tgz") or name.endswith(".tar.gz"):
        # Differentiate between Pub, Python sdist, binary bundles, and NPM packages
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                names = [m.name for m in tar.getmembers()]
                if any("pubspec.yaml" in n for n in names):
                    success, passed, errors = verify_pub_package(file_path)
                elif any(n.endswith("PKG-INFO") for n in names):
                    success, passed, errors = verify_sdist_package(file_path)
                elif any("package.json" in n for n in names):
                    success, passed, errors = verify_npm_package(file_path)
                elif "wasm" in name.lower() or "rusto-" in name.lower():
                    success, passed, errors = verify_archive_bundle(file_path)
                else:
                    success, passed, errors = verify_npm_package(file_path)
        except Exception as e:
            errors.append(f"Failed to open tarball: {e}")
    elif name.endswith(".nupkg"):
        success, passed, errors = verify_nuget_package(file_path)
    elif name.endswith(".aar"):
        success, passed, errors = verify_aar_package(file_path)
    elif name.endswith(".jar"):
        success, passed, errors = verify_jar_package(file_path)
    elif name.endswith(".whl"):
        success, passed, errors = verify_wheel_package(file_path)
    elif name.endswith(".zip") and "go" in name.lower():
        success, passed, errors = verify_go_package(file_path)
    elif name.endswith(".zip") and ("models" in name.lower()):
        success, passed, errors = verify_model_zip(file_path)
    elif name.endswith(".zip") and ("xcframework" in name.lower() or "rusto" in name.lower()):
        success, passed, errors = verify_xcframework_zip(file_path)
    else:
        print(f"⚠️ Unrecognized archive format: {file_path}. Skipping.")
        return True

    for p in passed:
        print(f"  ✅ {p}")
    for err in errors:
        print(f"  ❌ {err}")

    if success:
        print(f"✨ {name} VERIFIED SUCCESSFULLY.")
    else:
        print(f"💥 {name} FAILED CONTENT VERIFICATION.")

    return success

def scan_directory(dir_path: Path) -> bool:
    print(f"📂 Scanning directory for packed release packages: {dir_path}")
    all_success = True
    count = 0

    patterns = ["*.tgz", "*.tar.gz", "*.nupkg", "*.aar", "*.jar", "*.whl", "*xcframework*.zip", "*RustO*.zip", "*go*.zip"]
    matched_files: Set[Path] = set()

    for pattern in patterns:
        for f in dir_path.rglob(pattern):
            if f.is_file():
                matched_files.add(f)

    if not matched_files:
        print(f"⚠️ No packaged artifacts found in {dir_path}")
        return False

    for f in sorted(matched_files):
        count += 1
        if not verify_file(f):
            all_success = False

    print(f"\n=======================================================")
    print(f"Summary: Verified {count} package(s). Result: {'PASSED' if all_success else 'FAILED'}")
    print(f"=======================================================")
    return all_success

def main() -> int:
    parser = argparse.ArgumentParser(description="RustO Packed Package Content Verifier")
    parser.add_argument("--dir", type=Path, help="Directory to scan recursively for packages")
    parser.add_argument("--npm", type=Path, action="append", default=[], help="Path to NPM .tgz package to verify")
    parser.add_argument("--nuget", type=Path, action="append", default=[], help="Path to NuGet .nupkg package to verify")
    parser.add_argument("--aar", type=Path, action="append", default=[], help="Path to Android .aar package to verify")
    parser.add_argument("--jar", type=Path, action="append", default=[], help="Path to Java .jar package to verify")
    parser.add_argument("--pub", type=Path, action="append", default=[], help="Path to Dart/Flutter .tar.gz package to verify")
    parser.add_argument("--go", type=Path, action="append", default=[], help="Path to Go module directory or .zip to verify")
    parser.add_argument("--wheel", type=Path, action="append", default=[], help="Path to Python .whl package to verify")
    parser.add_argument("--xcframework", type=Path, action="append", default=[], help="Path to iOS .zip package to verify")
    parser.add_argument("--model-zip", type=Path, action="append", default=[], help="Path to model resource bundle .zip to verify")
    parser.add_argument("files", nargs="*", type=Path, help="Additional files or directories to verify")

    args = parser.parse_args()

    targets: List[Path] = []
    targets.extend(args.npm)
    targets.extend(args.nuget)
    targets.extend(args.aar)
    targets.extend(args.jar)
    targets.extend(args.pub)
    targets.extend(args.go)
    targets.extend(args.wheel)
    targets.extend(args.xcframework)
    targets.extend(args.model_zip)
    targets.extend(args.files)

    if args.dir:
        success = scan_directory(args.dir)
        return 0 if success else 1

    if not targets:
        # Default: scan current directory or build/releases
        for candidate in [Path("releases"), Path("build"), Path(".")]:
            if candidate.is_dir():
                print(f"No arguments provided. Scanning candidate directory: {candidate}")
                success = scan_directory(candidate)
                return 0 if success else 1
        parser.print_help()
        return 1

    all_passed = True
    for target in targets:
        if target.is_dir() and (target / "go.mod").exists():
            if not verify_file(target):
                all_passed = False
        elif target.is_dir():
            if not scan_directory(target):
                all_passed = False
        else:
            if not verify_file(target):
                all_passed = False

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
