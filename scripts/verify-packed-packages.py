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
            if not dll_entries and not any("models" in archive_path.name.lower() for _ in [1]):
                errors.append("Missing managed .dll assemblies in lib/")
            else:
                passed.append(f"Contains {len(dll_entries)} managed assembly/ies: {', '.join(dll_entries[:3])}")

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
            elif "model" not in archive_path.name.lower():
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

def verify_file(file_path: Path) -> bool:
    name = file_path.name
    print(f"\n🔍 Verifying: {file_path}")

    if not file_path.exists() or file_path.stat().st_size == 0:
        print(f"❌ Error: File does not exist or is 0 bytes: {file_path}")
        return False

    success = False
    passed: List[str] = []
    errors: List[str] = []

    if name.endswith(".tgz") or (name.endswith(".tar.gz") and "npm" in str(file_path)):
        success, passed, errors = verify_npm_package(file_path)
    elif name.endswith(".nupkg"):
        success, passed, errors = verify_nuget_package(file_path)
    elif name.endswith(".aar"):
        success, passed, errors = verify_aar_package(file_path)
    elif name.endswith(".whl"):
        success, passed, errors = verify_wheel_package(file_path)
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

    patterns = ["*.tgz", "*.nupkg", "*.aar", "*.whl", "*xcframework*.zip", "*RustO*.zip"]
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
    parser.add_argument("--wheel", type=Path, action="append", default=[], help="Path to Python .whl package to verify")
    parser.add_argument("--xcframework", type=Path, action="append", default=[], help="Path to iOS .zip package to verify")
    parser.add_argument("files", nargs="*", type=Path, help="Additional files or directories to verify")

    args = parser.parse_args()

    targets: List[Path] = []
    targets.extend(args.npm)
    targets.extend(args.nuget)
    targets.extend(args.aar)
    targets.extend(args.wheel)
    targets.extend(args.xcframework)
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
        if target.is_dir():
            if not scan_directory(target):
                all_passed = False
        else:
            if not verify_file(target):
                all_passed = False

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
