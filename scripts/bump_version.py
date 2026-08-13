#!/usr/bin/env python3
"""
RustO! Version Bump Script
Updates versions across all crates, native packages (Android, iOS, .NET), React Native, and documentation.
Automatically stages all updated versioned files in git.

Usage:
  python3 scripts/bump_version.py patch
  python3 scripts/bump_version.py minor
  python3 scripts/bump_version.py major
  python3 scripts/bump_version.py 0.1.4
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

def normalize_changelog_text(text: str) -> str:
    """
    Normalizes a changelog entry or commit message for deduplication comparison.
    Strips leading markdown list tokens, link syntax, bold/italic, scope prefixes,
    conventional commit prefixes, trailing issue/PR refs, whitespace, and punctuation.
    """
    text = re.sub(r"^[\s\-\*\+]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[\*_]{1,3}", "", text)
    text = re.sub(r"\s*\([#\d\w\s,-]+\)$", "", text)
    text = re.sub(r"^(feat|feature|fix|bug|patch|hotfix|chore|docs|refactor|style|test|perf|build|ci)(\([^)]+\))?:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.rstrip(".,;!?:")

def extract_existing_changelog_entries(changelog_path: Path, exclude_version: str = "") -> tuple:
    """
    Extracts all bullet points from existing sections in CHANGELOG.md,
    returning sets of normalized strings for deduplication:
    - existing: full normalized strings
    - core_existing: core text with any scope prefix stripped
    """
    existing: Set[str] = set()
    core_existing: Set[str] = set()
    if not changelog_path or not changelog_path.exists():
        return existing, core_existing

    try:
        with open(changelog_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return existing, core_existing

    sections = re.split(r"\n(?=##\s*\[)", content)
    for sec in sections:
        ver_match = re.match(r"##\s*\[([^\]]+)\]", sec.strip())
        if ver_match:
            sec_ver = ver_match.group(1).strip()
            if exclude_version and sec_ver == exclude_version:
                continue
        for line in sec.splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "+")):
                norm = normalize_changelog_text(line)
                if norm:
                    existing.add(norm)
                    core_norm = re.sub(r"^[a-z0-9_ -]+:\s*", "", norm)
                    if core_norm:
                        core_existing.add(core_norm)
    return existing, core_existing

def find_base_commit_for_changelog(current_ver: str) -> str:
    """
    Finds the base commit/tag from which to collect new changelog items.
    """
    # 1. Check if git tag exists for current_ver
    if current_ver:
        for tag in [f"v{current_ver}", current_ver]:
            res = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/tags/{tag}"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                return tag

    # 2. Check recent commit history for release commit of current_ver
    if current_ver:
        log_res = subprocess.run(
            ["git", "log", "--format=%H %s", "-n", "50"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True
        )
        pattern = rf"^(?:chore(?:\([^)]+\))?:\s*)?(?:release\s+)?v?{re.escape(current_ver)}$"
        for line in log_res.stdout.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commit_hash, subject = parts[0], parts[1].strip()
                if re.match(pattern, subject, re.IGNORECASE):
                    return commit_hash

    # 3. Fallback to latest tag in git
    try:
        tag_proc = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        latest_tag = tag_proc.stdout.strip()
        if latest_tag:
            return latest_tag
    except Exception:
        pass

    return ""

def get_commits_since_last_version(current_ver: str = "") -> List[str]:
    base_ref = find_base_commit_for_changelog(current_ver)
    git_range = f"{base_ref}..HEAD" if base_ref else "HEAD"

    try:
        log_proc = subprocess.run(
            ["git", "log", git_range, "--no-merges", "--pretty=format:%s"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return [l.strip() for l in log_proc.stdout.splitlines() if l.strip()]
    except Exception:
        return []

def generate_changelog_section(new_ver: str, current_ver: str = "", changelog_path: Path = None) -> str:
    import datetime
    commits = get_commits_since_last_version(current_ver)
    today = datetime.date.today().strftime("%Y-%m-%d")

    existing_entries: Set[str] = set()
    existing_core_entries: Set[str] = set()
    if changelog_path is not None:
        existing_entries, existing_core_entries = extract_existing_changelog_entries(changelog_path, exclude_version=new_ver)

    added: List[str] = []
    fixed: List[str] = []
    changed: List[str] = []
    seen_in_current: Set[str] = set()

    def add_item(target_list: List[str], item: str, raw_msg: str):
        norm = normalize_changelog_text(item)
        core_norm = re.sub(r"^[a-z0-9_ -]+:\s*", "", norm)
        raw_norm = normalize_changelog_text(raw_msg)
        raw_core_norm = re.sub(r"^[a-z0-9_ -]+:\s*", "", raw_norm)

        # Check deduplication against previous changelog entries and current section
        if norm in existing_entries or core_norm in existing_core_entries or norm in seen_in_current:
            return
        if raw_norm in existing_entries or raw_core_norm in existing_core_entries:
            return

        seen_in_current.add(norm)
        if core_norm:
            seen_in_current.add(core_norm)
        if item not in target_list:
            target_list.append(item)

    for msg in commits:
        if re.match(r"^(chore(\([^)]+\))?:\s*)?(release\s+)?v?\d+\.\d+\.\d+", msg, re.IGNORECASE):
            continue
        if re.match(r"^v?\d+\.\d+\.\d+$", msg):
            continue

        conv_match = re.match(r"^(\w+)(?:\(([^)]+)\))?:\s*(.+)$", msg, re.IGNORECASE)
        if conv_match:
            c_type = conv_match.group(1).lower()
            scope = conv_match.group(2)
            text = conv_match.group(3).strip()
            if text:
                text = text[0].upper() + text[1:]
            item = f"- **{scope}**: {text}" if scope else f"- {text}"

            if c_type in ("feat", "feature", "add"):
                add_item(added, item, msg)
            elif c_type in ("fix", "bug", "patch", "hotfix"):
                add_item(fixed, item, msg)
            else:
                add_item(changed, item, msg)
        else:
            clean_msg = msg.strip()
            if clean_msg:
                clean_msg = clean_msg[0].upper() + clean_msg[1:]
            lower = clean_msg.lower()
            if lower.startswith("add") or lower.startswith("feat"):
                add_item(added, f"- {clean_msg}", msg)
            elif lower.startswith("fix"):
                add_item(fixed, f"- {clean_msg}", msg)
            else:
                add_item(changed, f"- {clean_msg}", msg)

    section_lines = [f"## [{new_ver}] - {today}\n"]

    if added:
        section_lines.append("### Added")
        section_lines.extend(added)
        section_lines.append("")

    if changed:
        section_lines.append("### Changed")
        section_lines.extend(changed)
        section_lines.append("")

    if fixed:
        section_lines.append("### Fixed")
        section_lines.extend(fixed)
        section_lines.append("")

    if not added and not changed and not fixed:
        section_lines.append("### Changed")
        section_lines.append(f"- Maintenance release v{new_ver}")
        section_lines.append("")

    return "\n".join(section_lines).strip() + "\n\n"

def update_changelog_file(file_path: Path, new_ver: str, current_ver: str = "", dry_run: bool = False, staged_files: Set[Path] = None) -> bool:
    import datetime
    if not file_path.exists():
        print(f"  [WARN] File not found: {file_path.relative_to(REPO_ROOT)}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_section = generate_changelog_section(new_ver, current_ver=current_ver, changelog_path=file_path)

    # Check if section for new_ver already exists in CHANGELOG.md
    existing_pattern = rf"(##\s*\[{re.escape(new_ver)}\][^\n]*\n(?:(?!\n##\s*\[|\n---\n|\n##\s*Project\s*Status)[\s\S])*)"
    if re.search(existing_pattern, content):
        updated_content = re.sub(existing_pattern, new_section.strip(), content, count=1)
    else:
        first_section_match = re.search(r"\n(##\s*\[\d+\.\d+\.\d+\])", content)
        if first_section_match:
            pos = first_section_match.start(1)
            updated_content = content[:pos] + new_section + content[pos:]
        else:
            header_end = content.find("\n\n")
            if header_end != -1:
                updated_content = content[:header_end+2] + new_section + content[header_end+2:]
            else:
                updated_content = content + "\n\n" + new_section

    today_formatted = datetime.date.today().strftime("%B %d, %Y")
    updated_content = re.sub(
        r'(\*\*Version\*\*:\s*)[^\n]+',
        rf'\g<1>{new_ver}  ',
        updated_content
    )
    updated_content = re.sub(
        r'(\*\*Last Updated\*\*:\s*)[^\n]+',
        rf'\g<1>{today_formatted}',
        updated_content
    )

    if not dry_run and content != updated_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

    if staged_files is not None:
        staged_files.add(file_path)

    print(f"  [OK] Updated {file_path.relative_to(REPO_ROOT)} with new section for [{new_ver}]")
    return True

def update_all_files(new_ver: str, current_ver: str = "", dry_run: bool = False) -> Set[Path]:
    print(f"\nUpdating version to '{new_ver}' across repository files...")
    staged_files: Set[Path] = set()
    
    # 1. Cargo.toml (root package and mnn dependency)
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
    
    # 2. crates/mnn/Cargo.toml (package and mnn-sys dependency)
    replace_in_file(
        REPO_ROOT / "crates" / "mnn" / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    replace_in_file(
        REPO_ROOT / "crates" / "mnn" / "Cargo.toml",
        r'(mnn-sys\s*=\s*\{[^}]*version\s*=\s*")[^"]+(")',
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

    # 4. crates/mnn-sys/build.rs (default fallback version)
    replace_in_file(
        REPO_ROOT / "crates" / "mnn-sys" / "build.rs",
        r'(CARGO_PKG_VERSION"\)\.unwrap_or_else\(\|\_\s*\|\s*")[^"]+("\.to_string\(\)\))',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 5. packages/android/gradle.properties
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "gradle.properties",
        r'(VERSION_NAME\s*=\s*).*',
        rf'\g<1>{new_ver}',
        dry_run,
        staged_files
    )
    
    # 6. packages/android/**/build.gradle (root and all 20 model submodules)
    for bg_file in (REPO_ROOT / "packages" / "android").rglob("build.gradle"):
        replace_in_file(
            bg_file,
            r'(versionName\s+")[^"]+(")',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
        replace_in_file(
            bg_file,
            r"(coordinates\('[^']+',\s*'[^']+',\s*')[^']+(\'\))",
            rf"\g<1>{new_ver}\g<2>",
            dry_run,
            staged_files
        )
    
    # 7. packages/ios/**/*.podspec (binding package and every model subpackage)
    ios_dir = REPO_ROOT / "packages" / "ios"
    for podspec_file in ios_dir.rglob("*.podspec"):
        replace_in_file(
            podspec_file,
            r'(s\.version\s*=\s*\')[^\']+(\')',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    
    # 8. packages/**/package.json and sibling package-lock.json files.
    # Excludes vendored dependencies because search starts at first-party packages/.
    packages_dir = REPO_ROOT / "packages"
    for package_json_file in packages_dir.rglob("package.json"):
        update_json_file(package_json_file, new_ver, dry_run, staged_files)

        package_lock_file = package_json_file.with_name("package-lock.json")
        if package_lock_file.exists():
            update_json_file(package_lock_file, new_ver, dry_run, staged_files)
        
    # 10. packages/react-native/android/build.gradle
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
        rf"\g<1>v{new_ver}\g<2>",
        dry_run,
        staged_files
    )
    
    # 11. packages/dotnet/**/*.csproj (binding package and every model subpackage)
    dotnet_dir = REPO_ROOT / "packages" / "dotnet"
    for csproj_file in dotnet_dir.rglob("*.csproj"):
        replace_in_file(
            csproj_file,
            r'(<Version>)[^<]+(</Version>)',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    
    # 12. packages/dotnet/**/*.nuspec
    for nuspec_file in dotnet_dir.rglob("*.nuspec"):
        replace_in_file(
            nuspec_file,
            r'(<version>)[^<]+(</version>)',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    
    # 13. README.md & packages/react-native/README.md
    replace_in_file(
        REPO_ROOT / "README.md",
        r'(\*\*Version\*\*:\s*)[^\s]+',
        rf'\g<1>{new_ver}  ',
        dry_run,
        staged_files
    )
    replace_in_file(
        REPO_ROOT / "README.md",
        r'(com\.github\.byrizki\.rusto-rs:rusto-[^:]+:v)[0-9A-Za-z.-]+',
        rf'\g<1>{new_ver}',
        dry_run,
        staged_files
    )
    if (REPO_ROOT / "packages" / "react-native" / "README.md").exists():
        replace_in_file(
            REPO_ROOT / "packages" / "react-native" / "README.md",
            r'(com\.github\.byrizki\.rusto-rs:rusto-[^:]+:v)[0-9A-Za-z.-]+',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files
        )

    # 14. .github/workflows/publish.yml (default workflow_dispatch version)
    if (REPO_ROOT / ".github" / "workflows" / "publish.yml").exists():
        replace_in_file(
            REPO_ROOT / ".github" / "workflows" / "publish.yml",
            r'(version:\s*\n\s*description:[^\n]*\n\s*required:[^\n]*\n\s*default:\s*\')[^\']+(\')',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 15. CHANGELOG.md (auto-generated from git commits)
    if (REPO_ROOT / "CHANGELOG.md").exists():
        update_changelog_file(
            REPO_ROOT / "CHANGELOG.md",
            new_ver,
            current_ver=current_ver,
            dry_run=dry_run,
            staged_files=staged_files
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
  ./bump_version.sh patch          # 0.1.3 -> 0.1.4 (updates & stages files)
  ./bump_version.sh minor          # 0.1.3 -> 0.2.0 (updates & stages files)
  ./bump_version.sh major          # 0.1.3 -> 1.0.0 (updates & stages files)
  ./bump_version.sh 0.1.4          # Explicit version
  ./bump_version.sh --git patch    # Bump, stage, commit, and create git tag
  ./bump_version.sh --dry-run patch
  ./bump_version.sh --no-stage     # Update version without running git add
        """
    )
    parser.add_argument(
        "version",
        nargs="?",
        default="patch",
        help="Bump type ('patch', 'minor', 'major') or explicit version (e.g. '0.1.4'). Default: 'patch'"
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

    modified_files = update_all_files(new_ver, current_ver=current_ver, dry_run=args.dry_run)

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

