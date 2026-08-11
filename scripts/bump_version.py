#!/usr/bin/env python3
"""
RustO! Version Bump Script
Updates versions across all crates, native packages (Android, iOS, .NET), React Native, and documentation.
Automatically stages all updated versioned files in git.

Usage:
  python3 scripts/bump_version.py patch
  python3 scripts/bump_version.py minor
  python3 scripts/bump_version.py major
  python3 scripts/bump_version.py 0.1.3
  python3 scripts/bump_version.py --dry-run patch
  python3 scripts/bump_version.py --git patch
"""

import sys
import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent

def get_current_version() -> str:
    cargo_toml = REPO_ROOT / "Cargo.toml"
    with open(cargo_toml, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'\[package\]\s*\n(?:[^\[]*?\n)?version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not determine current version from {cargo_toml}")
    return match.group(1)

def parse_new_version(current_ver: str, bump_arg: str) -> str:
    semver_match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$", current_ver)
    if not semver_match:
        raise ValueError(f"Invalid current version format: '{current_ver}'")
    
    major, minor, patch = int(semver_match.group(1)), int(semver_match.group(2)), int(semver_match.group(3))
    
    if bump_arg == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_arg == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_arg == "major":
        return f"{major + 1}.0.0"
    else:
        # Explicit version provided
        if not re.match(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$", bump_arg):
            raise ValueError(f"Invalid version string: '{bump_arg}'. Must be semver like X.Y.Z or X.Y.Z-tag (or 'patch', 'minor', 'major').")
        return bump_arg

def replace_in_file(file_path: Path, pattern: str, replacement: str, dry_run: bool = False, staged_files: Set[Path] = None) -> bool:
    if not file_path.exists():
        print(f"  [WARN] File not found: {file_path.relative_to(REPO_ROOT)}")
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        original = f.read()
        
    modified, count = re.subn(pattern, replacement, original)
    if count == 0:
        print(f"  [WARN] Pattern not matched in {file_path.relative_to(REPO_ROOT)}")
        return False
    
    if not dry_run and original != modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified)
    
    if staged_files is not None:
        staged_files.add(file_path)
    
    print(f"  [OK] Updated {count} occurrence(s) in {file_path.relative_to(REPO_ROOT)}")
    return True

def update_json_file(file_path: Path, new_ver: str, dry_run: bool = False, staged_files: Set[Path] = None) -> bool:
    if not file_path.exists():
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["version"] = new_ver
    
    # Handle package-lock.json v2/v3 structure
    if "packages" in data and "" in data["packages"]:
        data["packages"][""]["version"] = new_ver
        
    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            
    if staged_files is not None:
        staged_files.add(file_path)
        
    print(f"  [OK] Updated version to {new_ver} in {file_path.relative_to(REPO_ROOT)}")
    return True

def update_all_files(new_ver: str, dry_run: bool = False) -> Set[Path]:
    print(f"\nUpdating version to '{new_ver}' across repository files...")
    staged_files: Set[Path] = set()
    
    # 1. Cargo.toml (root)
    replace_in_file(
        REPO_ROOT / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    replace_in_file(
        REPO_ROOT / "Cargo.toml",
        r'(mnn\s*=\s*\{[^}]*version\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 2. crates/mnn/Cargo.toml
    replace_in_file(
        REPO_ROOT / "crates" / "mnn" / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 3. crates/mnn-sys/Cargo.toml
    replace_in_file(
        REPO_ROOT / "crates" / "mnn-sys" / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 4. packages/android/gradle.properties
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "gradle.properties",
        r'(VERSION_NAME\s*=\s*).*',
        rf'\g<1>{new_ver}',
        dry_run,
        staged_files
    )
    
    # 5. packages/android/build.gradle
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "build.gradle",
        r'(versionName\s+")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "build.gradle",
        r'(version\s*=\s*\')[^\']+(\')',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 6. packages/ios/RustO.podspec
    replace_in_file(
        REPO_ROOT / "packages" / "ios" / "RustO.podspec",
        r'(s\.version\s*=\s*\')[^\']+(\')',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 7. packages/react-native/package.json
    update_json_file(
        REPO_ROOT / "packages" / "react-native" / "package.json",
        new_ver,
        dry_run,
        staged_files
    )
    
    # 8. packages/react-native/package-lock.json
    if (REPO_ROOT / "packages" / "react-native" / "package-lock.json").exists():
        update_json_file(
            REPO_ROOT / "packages" / "react-native" / "package-lock.json",
            new_ver,
            dry_run,
            staged_files
        )
        
    # 9. packages/react-native/android/build.gradle
    replace_in_file(
        REPO_ROOT / "packages" / "react-native" / "android" / "build.gradle",
        r'(versionName\s+")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    replace_in_file(
        REPO_ROOT / "packages" / "react-native" / "android" / "build.gradle",
        r"(safeExtGet\('RustoAndroid_version',\s*')[^']+(\'\))",
        rf"\g<1>{new_ver}\g<2>",
        dry_run,
        staged_files
    )
    
    # 10. packages/dotnet/RustODotnet.csproj
    replace_in_file(
        REPO_ROOT / "packages" / "dotnet" / "RustODotnet.csproj",
        r'(<Version>)[^<]+(</Version>)',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 11. packages/dotnet/RustODotnet.nuspec
    replace_in_file(
        REPO_ROOT / "packages" / "dotnet" / "RustODotnet.nuspec",
        r'(<version>)[^<]+(</version>)',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 12. README.md
    replace_in_file(
        REPO_ROOT / "README.md",
        r'(\*\*Version\*\*:\s*)[^\s]+',
        rf'\g<1>{new_ver}  ',
        dry_run,
        staged_files
    )

    return staged_files

def stage_git_files(files: Set[Path]):
    if not files:
        return
    file_rel_paths = [str(f.relative_to(REPO_ROOT)) for f in sorted(files) if f.exists()]
    print(f"\nAuto-staging {len(file_rel_paths)} versioned files in git:")
    for rel_path in file_rel_paths:
        print(f"  + git add {rel_path}")
    try:
        subprocess.run(["git", "add"] + file_rel_paths, cwd=REPO_ROOT, check=True)
        print("  [OK] Successfully staged versioned files in git.")
    except Exception as e:
        print(f"  [WARN] Failed to run git add: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Bump version across all packages in the RustO! monorepo and auto-stage files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./bump_version.sh patch          # 0.1.2 -> 0.1.3 (updates & stages files)
  ./bump_version.sh minor          # 0.1.2 -> 0.2.0 (updates & stages files)
  ./bump_version.sh major          # 0.1.2 -> 1.0.0 (updates & stages files)
  ./bump_version.sh 0.1.3          # Explicit version
  ./bump_version.sh --git patch    # Bump, stage, commit, and create git tag
  ./bump_version.sh --no-stage     # Update version without running git add
        """
    )
    parser.add_argument(
        "version",
        nargs="?",
        default="patch",
        help="Bump type ('patch', 'minor', 'major') or explicit version (e.g. '0.1.3'). Default: 'patch'"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the bump without modifying files"
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Do not automatically stage modified files with git add"
    )
    parser.add_argument(
        "-g", "--git",
        action="store_true",
        help="Create a git commit and annotated tag for the new version"
    )

    args = parser.parse_args()

    try:
        current_ver = get_current_version()
        new_ver = parse_new_version(current_ver, args.version)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Current version : {current_ver}")
    print(f"New version     : {new_ver}")
    if args.dry_run:
        print("Mode            : [DRY-RUN - No files will be modified]")

    modified_files = update_all_files(new_ver, args.dry_run)

    # Sync Cargo.lock if cargo is installed
    if not args.dry_run:
        print("\nUpdating Cargo.lock...")
        try:
            subprocess.run(
                ["cargo", "check", "--workspace"],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            cargo_lock = REPO_ROOT / "Cargo.lock"
            if cargo_lock.exists():
                modified_files.add(cargo_lock)
            print("  [OK] Cargo.lock updated.")
        except Exception:
            pass

        # Auto stage modified versioned files
        if not args.no_stage:
            stage_git_files(modified_files)

    print(f"\n✨ Version bump to v{new_ver} complete!")

    if args.git and not args.dry_run:
        print("\nCreating Git commit and tag...")
        commit_msg = f"chore: release v{new_ver}"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
        tag_name = f"v{new_ver}"
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"], cwd=REPO_ROOT, check=True)
        print(f"  [OK] Created commit: '{commit_msg}'")
        print(f"  [OK] Created git tag: '{tag_name}'")
        print("\nTo push changes and trigger CI/CD release:")
        print(f"  git push origin main --tags")
    elif not args.dry_run:
        print("\nSuggested next steps:")
        print(f"  git commit -m \"chore: release v{new_ver}\"")
        print(f"  git tag -a v{new_ver} -m \"Release v{new_ver}\"")
        print(f"  git push origin main --tags")

if __name__ == "__main__":
    main()
