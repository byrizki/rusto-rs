#!/usr/bin/env python3
"""
RustO! Version Bump Script
Updates versions across all crates, publishable packages, binding consumer examples, workflows, and documentation.
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
from typing import List, Set, Dict, Tuple, NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

def get_current_lib_version() -> str:
    cargo_toml = REPO_ROOT / "Cargo.toml"
    with open(cargo_toml, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'\[package\]\s*\n(?:[^\[]*?\n)?version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not determine current version from {cargo_toml}")
    return match.group(1)

def get_current_models_version() -> str:
    models_file = REPO_ROOT / "MODELS_VERSION"
    if models_file.exists():
        val = models_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    return get_current_lib_version()

def get_current_version(target: str = "lib") -> str:
    if target == "models":
        return get_current_models_version()
    return get_current_lib_version()

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

def replace_in_file(file_path: Path, pattern: str, replacement: str, dry_run: bool = False, staged_files: Set[Path] = None, warn_if_not_found: bool = True) -> bool:
    if not file_path.exists():
        if warn_if_not_found:
            print(f"  [WARN] File not found: {file_path.relative_to(REPO_ROOT)}")
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        original = f.read()
        
    modified, count = re.subn(pattern, replacement, original)
    if count == 0:
        if warn_if_not_found:
            print(f"  [WARN] Pattern not matched in {file_path.relative_to(REPO_ROOT)}")
        return False
    
    if not dry_run and original != modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified)
    
    if staged_files is not None:
        staged_files.add(file_path)
    
    print(f"  [OK] Updated {count} occurrence(s) in {file_path.relative_to(REPO_ROOT)}")
    return True

def update_pub_changelog(file_path: Path, new_ver: str, dry_run: bool = False, staged_files: Set[Path] = None) -> bool:
    if not file_path.exists():
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if re.search(rf"(?m)^##\s+{re.escape(new_ver)}\b", content):
        return True

    new_section = f"## {new_ver}\n\n- Update RustO native library and bindings to {new_ver}.\n\n"
    first_sec = re.search(r"(?m)^##\s+", content)
    if first_sec:
        pos = first_sec.start()
        updated = content[:pos] + new_section + content[pos:]
    else:
        updated = content.rstrip() + "\n\n" + new_section

    if not dry_run and content != updated:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)

    if staged_files is not None:
        staged_files.add(file_path)

    print(f"  [OK] Updated changelog section ## {new_ver} in {file_path.relative_to(REPO_ROOT)}")
    return True

def update_json_file(file_path: Path, new_ver: str, dry_run: bool = False, staged_files: Set[Path] = None) -> bool:
    if not file_path.exists():
        return False
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "version" not in data:
        return False

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
    conventional commit prefixes, trailing committer/user tags, issue/PR refs, whitespace, and punctuation.
    """
    text = re.sub(r"^[\s\-\*\+]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[\*_]{1,3}", "", text)
    text = re.sub(r"\s+by\s+@[\w-]+(?:\[bot\])?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([#@\d\w\s,-]+(?:\[bot\])?\)$", "", text)
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

class CommitInfo(NamedTuple):
    hash: str
    subject: str
    committer_name: str
    committer_email: str
    author_name: str
    author_email: str
    github_user: str = ""

    def __str__(self) -> str:
        return self.subject

def get_github_repo_slug() -> str:
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        url = proc.stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", url)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    except Exception:
        pass
    return "byrizki/rusto-rs"

def resolve_commit_github_users(commits: List[CommitInfo]) -> List[CommitInfo]:
    if not commits:
        return []

    repo_slug = get_github_repo_slug()
    sha_map: Dict[str, str] = {}
    email_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}

    # Seed current authenticated gh user & local git config
    try:
        gh_user_proc = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        current_gh_user = gh_user_proc.stdout.strip()
        if current_gh_user:
            git_email_proc = subprocess.run(
                ["git", "config", "user.email"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            local_email = git_email_proc.stdout.strip().lower()
            if local_email:
                email_map[local_email] = current_gh_user
            git_name_proc = subprocess.run(
                ["git", "config", "user.name"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            local_name = git_name_proc.stdout.strip().lower()
            if local_name:
                name_map[local_name] = current_gh_user
    except Exception:
        pass

    # Batch query GitHub API for recent commits in repo
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo_slug}/commits?per_page=100"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            for item in data:
                sha = item.get("sha")
                author = item.get("author") or {}
                committer = item.get("committer") or {}
                a_login = author.get("login")
                c_login = committer.get("login")

                # Prefer committer login; when web-flow/github merge bot is committer, fallback to author login
                login = c_login if c_login and c_login not in ("web-flow", "github") else a_login
                if sha and login:
                    sha_map[sha] = login

                commit_meta = item.get("commit") or {}
                c_email = (commit_meta.get("committer") or {}).get("email")
                c_name = (commit_meta.get("committer") or {}).get("name")
                a_email = (commit_meta.get("author") or {}).get("email")
                a_name = (commit_meta.get("author") or {}).get("name")

                if c_email and c_login and c_login not in ("web-flow", "github"):
                    email_map[c_email.lower()] = c_login
                if a_email and a_login:
                    email_map[a_email.lower()] = a_login
                if c_name and c_login and c_login not in ("web-flow", "github"):
                    name_map[c_name.lower()] = c_login
                if a_name and a_login:
                    name_map[a_name.lower()] = a_login
    except Exception:
        pass

    resolved: List[CommitInfo] = []
    for c in commits:
        user = ""
        # 1. Exact SHA
        if c.hash and c.hash in sha_map:
            user = sha_map[c.hash]

        # 2. Committer email
        if not user and c.committer_email and c.committer_email.lower() in email_map:
            user = email_map[c.committer_email.lower()]

        # 3. Author email
        if not user and c.author_email and c.author_email.lower() in email_map:
            user = email_map[c.author_email.lower()]

        # 4. GitHub noreply email regex
        if not user:
            for email in (c.committer_email, c.author_email):
                if email:
                    m = re.match(r"^(?:[0-9]+\+)?([^@]+)@users\.noreply\.github\.com$", email)
                    if m:
                        matched_user = m.group(1)
                        if matched_user not in ("web-flow", "github"):
                            user = matched_user
                            break

        # 5. Committer / Author name in mapping
        if not user and c.committer_name and c.committer_name.lower() in name_map:
            user = name_map[c.committer_name.lower()]
        if not user and c.author_name and c.author_name.lower() in name_map:
            user = name_map[c.author_name.lower()]

        # 6. Fallback single commit API if SHA exists and not found yet
        if not user and c.hash:
            try:
                single_proc = subprocess.run(
                    ["gh", "api", f"repos/{repo_slug}/commits/{c.hash}"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if single_proc.returncode == 0 and single_proc.stdout.strip():
                    single_data = json.loads(single_proc.stdout)
                    c_login = (single_data.get("committer") or {}).get("login")
                    a_login = (single_data.get("author") or {}).get("login")
                    login = c_login if c_login and c_login not in ("web-flow", "github") else a_login
                    if login:
                        user = login
                        sha_map[c.hash] = login
            except Exception:
                pass

        # 7. Fallback if name is a valid single-word GitHub username
        if not user:
            for name in (c.committer_name, c.author_name):
                if name and re.match(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$", name):
                    user = name
                    break

        resolved.append(CommitInfo(
            hash=c.hash,
            subject=c.subject,
            committer_name=c.committer_name,
            committer_email=c.committer_email,
            author_name=c.author_name,
            author_email=c.author_email,
            github_user=user,
        ))

    return resolved

def get_commits_since_last_version(current_ver: str = "") -> List[CommitInfo]:
    base_ref = find_base_commit_for_changelog(current_ver)
    git_range = f"{base_ref}..HEAD" if base_ref else "HEAD"

    try:
        log_proc = subprocess.run(
            ["git", "log", git_range, "--no-merges", "--pretty=format:%H%x09%s%x09%cn%x09%ce%x09%an%x09%ae"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        raw_lines = [l.strip() for l in log_proc.stdout.splitlines() if l.strip()]
    except Exception:
        return []

    commits: List[CommitInfo] = []
    for line in raw_lines:
        parts = line.split("\t")
        if len(parts) >= 6:
            commits.append(CommitInfo(
                hash=parts[0].strip(),
                subject=parts[1].strip(),
                committer_name=parts[2].strip(),
                committer_email=parts[3].strip(),
                author_name=parts[4].strip(),
                author_email=parts[5].strip(),
            ))
        elif len(parts) >= 2:
            commits.append(CommitInfo(
                hash=parts[0].strip(),
                subject=parts[1].strip(),
                committer_name="",
                committer_email="",
                author_name="",
                author_email="",
            ))
        elif parts:
            commits.append(CommitInfo(
                hash="",
                subject=parts[0].strip(),
                committer_name="",
                committer_email="",
                author_name="",
                author_email="",
            ))

    return resolve_commit_github_users(commits)

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

    for commit in commits:
        msg = commit.subject if isinstance(commit, CommitInfo) else str(commit)
        user = commit.github_user if isinstance(commit, CommitInfo) else ""

        if re.match(r"^(chore(\([^)]+\))?:\s*)?(release\s+)?v?\d+\.\d+\.\d+", msg, re.IGNORECASE):
            continue
        if re.match(r"^v?\d+\.\d+\.\d+$", msg):
            continue

        clean_user_pattern = rf"\s*(?:\(@{re.escape(user)}\)|by\s+@{re.escape(user)})\s*$" if user else None
        user_tag = f" (@{user})" if user else ""

        conv_match = re.match(r"^(\w+)(?:\(([^)]+)\))?:\s*(.+)$", msg, re.IGNORECASE)
        if conv_match:
            c_type = conv_match.group(1).lower()
            scope = conv_match.group(2)
            text = conv_match.group(3).strip()
            if clean_user_pattern:
                text = re.sub(clean_user_pattern, "", text, flags=re.IGNORECASE).strip()
            if text:
                text = text[0].upper() + text[1:]
            item = f"- **{scope}**: {text}{user_tag}" if scope else f"- {text}{user_tag}"

            if c_type in ("feat", "feature", "add"):
                add_item(added, item, msg)
            elif c_type in ("fix", "bug", "patch", "hotfix"):
                add_item(fixed, item, msg)
            else:
                add_item(changed, item, msg)
        else:
            clean_msg = msg.strip()
            if clean_user_pattern:
                clean_msg = re.sub(clean_user_pattern, "", clean_msg, flags=re.IGNORECASE).strip()
            if clean_msg:
                clean_msg = clean_msg[0].upper() + clean_msg[1:]
            lower = clean_msg.lower()
            if lower.startswith("add") or lower.startswith("feat"):
                add_item(added, f"- {clean_msg}{user_tag}", msg)
            elif lower.startswith("fix"):
                add_item(fixed, f"- {clean_msg}{user_tag}", msg)
            else:
                add_item(changed, f"- {clean_msg}{user_tag}", msg)

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

def extract_changelog_section(version: str, changelog_path: Path = None) -> str:
    ver = version.lstrip("v")
    if changelog_path is None:
        changelog_path = REPO_ROOT / "CHANGELOG.md"

    if changelog_path.exists():
        content = changelog_path.read_text(encoding="utf-8")
        pattern = rf"(?:^|\n)(##\s*\[{re.escape(ver)}\][^\n]*\n(?:(?!\n##\s*\[|\n---\n|\n##\s*Project\s*Status)[\s\S])*)"
        m = re.search(pattern, content)
        if m:
            return m.group(1).strip()

    return generate_changelog_section(ver, changelog_path=changelog_path).strip()

def update_lib_files(new_ver: str, current_ver: str = "", dry_run: bool = False) -> Set[Path]:
    print(f"\nUpdating library version to '{new_ver}' across repository files...")
    staged_files: Set[Path] = set()
    
    # 1. Cargo.toml (root package)
    replace_in_file(
        REPO_ROOT / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )

    # 2. crates/rten-wasm/Cargo.toml
    replace_in_file(
        REPO_ROOT / "crates" / "rten-wasm" / "Cargo.toml",
        r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )

    # 3. crates/rusto-node/Cargo.toml
    rusto_node_cargo = REPO_ROOT / "crates" / "rusto-node" / "Cargo.toml"
    if rusto_node_cargo.exists():
        replace_in_file(
            rusto_node_cargo,
            r'(?ms)(\[package\](?:(?!\[).)*?\bversion\s*=\s*")[^"]+(")',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    
    # 4. packages/android/gradle.properties (VERSION_NAME only, not MODELS_VERSION_NAME)
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "gradle.properties",
        r'(?m)^(VERSION_NAME\s*=\s*).*',
        rf'\g<1>{new_ver}',
        dry_run,
        staged_files
    )
    
    # 5. packages/android/build.gradle (root core library only)
    root_bg = REPO_ROOT / "packages" / "android" / "build.gradle"
    replace_in_file(
        root_bg,
        r'(versionName\s+")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    replace_in_file(
        root_bg,
        r"(coordinates\('com\.github\.byrizki\.rusto-rs',\s*'rusto-android',\s*')[^']+(\'\))",
        rf"\g<1>{new_ver}\g<2>",
        dry_run,
        staged_files
    )
    
    # 6. packages/ios/RustO.podspec (core binding podspec only)
    replace_in_file(
        REPO_ROOT / "packages" / "ios" / "RustO.podspec",
        r'(s\.version\s*=\s*\')[^\']+(\')',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 7. First-party npm library manifests and consumer examples
    for pkg_dir in [
        REPO_ROOT / "packages" / "web",
        REPO_ROOT / "packages" / "react",
        REPO_ROOT / "packages" / "react-native",
        REPO_ROOT / "packages" / "node",
        REPO_ROOT / "packages" / "react-native" / "example",
        REPO_ROOT / "examples" / "react-native-legacy",
        REPO_ROOT / "examples" / "react-native-new-arch",
    ]:
        pkg_json = pkg_dir / "package.json"
        if pkg_json.exists():
            update_json_file(pkg_json, new_ver, dry_run, staged_files)

    # 8. packages/react-native/android/build.gradle
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

    # 9. packages/python/pyproject.toml
    pyproject_file = REPO_ROOT / "packages" / "python" / "pyproject.toml"
    if pyproject_file.exists():
        replace_in_file(
            pyproject_file,
            r'(?m)^(version\s*=\s*")[^"]+(")',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 10. packages/dart/pubspec.yaml & CHANGELOG.md
    dart_pubspec = REPO_ROOT / "packages" / "dart" / "pubspec.yaml"
    if dart_pubspec.exists():
        replace_in_file(
            dart_pubspec,
            r'(?m)^(version:\s*).*',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files
        )
        update_pub_changelog(
            REPO_ROOT / "packages" / "dart" / "CHANGELOG.md",
            new_ver,
            dry_run,
            staged_files
        )

    # 11. packages/flutter/pubspec.yaml & CHANGELOG.md
    flutter_pubspec = REPO_ROOT / "packages" / "flutter" / "pubspec.yaml"
    if flutter_pubspec.exists():
        replace_in_file(
            flutter_pubspec,
            r'(?m)^(version:\s*).*',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files
        )
        replace_in_file(
            flutter_pubspec,
            r'(?m)^(\s*rusto:\s*)\^[0-9A-Za-z.-]+',
            rf'\g<1>^{new_ver}',
            dry_run,
            staged_files,
            warn_if_not_found=False
        )
        update_pub_changelog(
            REPO_ROOT / "packages" / "flutter" / "CHANGELOG.md",
            new_ver,
            dry_run,
            staged_files
        )

    # 12. packages/java/build.gradle & packages/java/README.md
    java_build = REPO_ROOT / "packages" / "java" / "build.gradle"
    if java_build.exists():
        replace_in_file(
            java_build,
            r'(?m)^(version\s*=\s*[\'"])[^\'"]+([\'"])',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    java_readme = REPO_ROOT / "packages" / "java" / "README.md"
    if java_readme.exists():
        replace_in_file(
            java_readme,
            r'(com\.github\.byrizki(?:\.rusto-rs)?:rusto-java:)[0-9A-Za-z.-]+',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files
        )
        replace_in_file(
            java_readme,
            r'(<artifactId>rusto-java</artifactId>\s*<version>)[^<]+(</version>)',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )
    
    # 13. packages/dotnet/RustODotnet.csproj (core package only)
    replace_in_file(
        REPO_ROOT / "packages" / "dotnet" / "RustODotnet.csproj",
        r'(<Version>)[^<]+(</Version>)',
        rf'\g<1>{new_ver}\g<2>',
        dry_run,
        staged_files
    )
    
    # 14. packages/dotnet/*.nuspec (core only)
    for nuspec_file in (REPO_ROOT / "packages" / "dotnet").glob("*.nuspec"):
        replace_in_file(
            nuspec_file,
            r'(<version>)[^<]+(</version>)',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 15. Consumer examples reference RustODotnet
    examples_dotnet_dir = REPO_ROOT / "examples" / "dotnet"
    if examples_dotnet_dir.exists():
        for csproj_file in examples_dotnet_dir.rglob("*.csproj"):
            replace_in_file(
                csproj_file,
                r'(<PackageReference\s+Include="RustODotnet"\s+Version=")[^"]+("\s*/?>)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files
            )
    
    # 16. README.md & packages/react-native/README.md
    readme_file = REPO_ROOT / "README.md"
    if readme_file.exists():
        replace_in_file(
            readme_file,
            r'(\*\*Version\*\*:\s*)[^\s]+',
            rf'\g<1>{new_ver}  ',
            dry_run,
            staged_files,
            warn_if_not_found=False
        )
        replace_in_file(
            readme_file,
            r'(com\.github\.byrizki\.rusto-rs:rusto-[^:]+:v)[0-9A-Za-z.-]+',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files,
            warn_if_not_found=False
        )
    rn_readme = REPO_ROOT / "packages" / "react-native" / "README.md"
    if rn_readme.exists():
        replace_in_file(
            rn_readme,
            r'(com\.github\.byrizki\.rusto-rs:rusto-[^:]+:v)[0-9A-Za-z.-]+',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files,
            warn_if_not_found=False
        )

    # 17. .github/workflows/publish.yml default version
    if (REPO_ROOT / ".github" / "workflows" / "publish.yml").exists():
        replace_in_file(
            REPO_ROOT / ".github" / "workflows" / "publish.yml",
            r'(version:\s*\n\s*description:[^\n]*\n\s*required:[^\n]*\n\s*default:\s*\')[^\']+(\')',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 18. Documentation guides across all supported languages
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        for doc_file in docs_dir.rglob("2.installation.md"):
            replace_in_file(
                doc_file,
                r'(rusto-rs\s*=\s*")[^"]+(")',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files
            )
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki(?:\.rusto-rs)?:rusto-java:)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(<artifactId>rusto-java</artifactId>\s*<version>)[^<]+(</version>)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(\.package\(url:\s*"https://github\.com/byrizki/rusto-rs\.git",\s*from:\s*")[^"]+(")',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-android:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        for doc_file in docs_dir.rglob("*00.use-rust.md"):
            replace_in_file(
                doc_file,
                r'(rusto-rs\s*=\s*")[^"]+(")',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        for doc_file in docs_dir.rglob("*03.use-ios-swift.md"):
            replace_in_file(
                doc_file,
                r'(\.package\(url:\s*"https://github\.com/byrizki/rusto-rs\.git",\s*from:\s*")[^"]+(")',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(pod\s+\'RustO\',\s*\'~>\s*)[0-9A-Za-z.-]+(\')',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(\(e\.g\.,\s*`?)[0-9]+\.[0-9]+\.[0-9]+(`?\))',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(\(misalnya\s*`?)[0-9]+\.[0-9]+\.[0-9]+(`?\))',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        for doc_file in docs_dir.rglob("*12.use-java.md"):
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki(?:\.rusto-rs)?:rusto-java:)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(<artifactId>rusto-java</artifactId>\s*<version>)[^<]+(</version>)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        for doc_file in docs_dir.rglob("*04.use-android-kotlin.md"):
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-android:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

    # 19. CHANGELOG.md (auto-generated from git commits)
    if (REPO_ROOT / "CHANGELOG.md").exists():
        update_changelog_file(
            REPO_ROOT / "CHANGELOG.md",
            new_ver,
            current_ver=current_ver,
            dry_run=dry_run,
            staged_files=staged_files
        )

    return staged_files

def update_models_files(new_ver: str, current_ver: str = "", dry_run: bool = False) -> Set[Path]:
    print(f"\nUpdating OCR models version to '{new_ver}' across repository files...")
    staged_files: Set[Path] = set()

    # 1. MODELS_VERSION file
    models_ver_file = REPO_ROOT / "MODELS_VERSION"
    if not dry_run:
        models_ver_file.write_text(f"{new_ver}\n", encoding="utf-8")
    if staged_files is not None:
        staged_files.add(models_ver_file)
    print(f"  [OK] Updated MODELS_VERSION to {new_ver}")

    # 2. packages/android/gradle.properties (MODELS_VERSION_NAME)
    replace_in_file(
        REPO_ROOT / "packages" / "android" / "gradle.properties",
        r'(?m)^(MODELS_VERSION_NAME\s*=\s*).*',
        rf'\g<1>{new_ver}',
        dry_run,
        staged_files
    )

    # 3. packages/android/models/**/build.gradle (all 20+ model submodules)
    android_models_dir = REPO_ROOT / "packages" / "android" / "models"
    if android_models_dir.exists():
        for bg_file in android_models_dir.rglob("build.gradle"):
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

    # 4. packages/ios/models/*.podspec (all 20+ model podspecs)
    ios_models_dir = REPO_ROOT / "packages" / "ios" / "models"
    if ios_models_dir.exists():
        for podspec_file in ios_models_dir.glob("*.podspec"):
            replace_in_file(
                podspec_file,
                r'(s\.version\s*=\s*\')[^\']+(\')',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files
            )

    # 5. packages/dotnet/models/**/*.csproj (all 20+ model projects)
    dotnet_models_dir = REPO_ROOT / "packages" / "dotnet" / "models"
    if dotnet_models_dir.exists():
        for csproj_file in dotnet_models_dir.rglob("*.csproj"):
            replace_in_file(
                csproj_file,
                r'(<Version>)[^<]+(</Version>)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files
            )

    # 6. examples/dotnet/Directory.Build.props (RustOArtifactVersion fallback for models)
    dotnet_props = REPO_ROOT / "examples" / "dotnet" / "Directory.Build.props"
    if dotnet_props.exists():
        replace_in_file(
            dotnet_props,
            r'(<RustOArtifactVersion Condition="\'\$\(RustOArtifactVersion\)\' == \'\'">)[^<]+(</RustOArtifactVersion>)',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 7. packages/models/**/package.json & README.md (all npm model packages)
    packages_models_dir = REPO_ROOT / "packages" / "models"
    if packages_models_dir.exists():
        for package_json_file in packages_models_dir.rglob("package.json"):
            if "node_modules" in package_json_file.parts:
                continue
            update_json_file(package_json_file, new_ver, dry_run, staged_files)
        for model_readme in packages_models_dir.rglob("README.md"):
            if "node_modules" in model_readme.parts:
                continue
            replace_in_file(
                model_readme,
                r'(- \*\*Version:\*\*\s*`)[^`]+(`)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files
            )

    # 8. scripts/generate_npm_models.py (fallback return)
    if (REPO_ROOT / "scripts" / "generate_npm_models.py").exists():
        replace_in_file(
            REPO_ROOT / "scripts" / "generate_npm_models.py",
            r'(return\s+")[0-9A-Za-z.-]+(")',
            rf'\g<1>{new_ver}\g<2>',
            dry_run,
            staged_files
        )

    # 9. packages/react-native/README.md (model dependency example)
    rn_readme = REPO_ROOT / "packages" / "react-native" / "README.md"
    if rn_readme.exists():
        replace_in_file(
            rn_readme,
            r'(com\.github\.byrizki\.rusto-rs:rusto-models-[^:]+:v?)[0-9A-Za-z.-]+',
            rf'\g<1>{new_ver}',
            dry_run,
            staged_files,
            warn_if_not_found=False
        )

    # 10. Documentation guides for models
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        # Installation docs (2.installation.md)
        for doc_file in docs_dir.rglob("2.installation.md"):
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-models-[^:]+:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        # React Native docs (02.use-react-native.md)
        for doc_file in docs_dir.rglob("*02.use-react-native.md"):
            replace_in_file(
                doc_file,
                r'(pod\s+\'RustO-Models-[^\']+\',\s*\'~>\s*)[0-9A-Za-z.-]+(\')',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-models-[^:]+:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        # iOS Swift docs (03.use-ios-swift.md)
        for doc_file in docs_dir.rglob("*03.use-ios-swift.md"):
            replace_in_file(
                doc_file,
                r'(pod\s+\'RustO-Models-[^\']+\',\s*\'~>\s*)[0-9A-Za-z.-]+(\')',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        # Android docs (04.use-android-kotlin.md)
        for doc_file in docs_dir.rglob("*04.use-android-kotlin.md"):
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-models-[^:]+:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        # Models & Tiers docs (01.models-and-tiers.md)
        for doc_file in docs_dir.rglob("*01.models-and-tiers.md"):
            replace_in_file(
                doc_file,
                r'(pod\s+\'RustO-Models-[^\']+\',\s*\'~>\s*)[0-9A-Za-z.-]+(\')',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(com\.github\.byrizki\.rusto-rs:rusto-models-[^:]+:v?)[0-9A-Za-z.-]+',
                rf'\g<1>{new_ver}',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )
            replace_in_file(
                doc_file,
                r'(https://github\.com/byrizki/rusto-rs/releases/download/models-rten-v)[0-9A-Za-z.-]+(/)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

        # Multilingual OCR docs (02.multilingual-ocr.md)
        for doc_file in docs_dir.rglob("*02.multilingual-ocr.md"):
            replace_in_file(
                doc_file,
                r'(https://github\.com/byrizki/rusto-rs/releases/download/models-rten-v)[0-9A-Za-z.-]+(/)',
                rf'\g<1>{new_ver}\g<2>',
                dry_run,
                staged_files,
                warn_if_not_found=False
            )

    return staged_files

def update_all_files(new_ver: str, current_ver: str = "", dry_run: bool = False) -> Set[Path]:
    staged = set()
    staged.update(update_lib_files(new_ver, current_ver=current_ver, dry_run=dry_run))
    staged.update(update_models_files(new_ver, current_ver=current_ver, dry_run=dry_run))
    return staged

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
        description="Bump version across packages in the RustO! monorepo and auto-stage files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./bump_version.sh patch                 # 0.3.0 -> 0.3.1 (updates lib packages only)
  ./bump_version.sh minor                 # 0.3.0 -> 0.4.0 (updates lib packages only)
  ./bump_version.sh --models patch        # 0.3.0 -> 0.3.1 (updates OCR model packages only)
  ./bump_version.sh --all patch           # 0.3.0 -> 0.3.1 (updates both lib and model packages)
  ./bump_version.sh --git patch           # Bump lib, stage, commit, and tag 'v0.3.1'
  ./bump_version.sh --models --git patch  # Bump models, stage, commit, and tag 'models-rten-v0.3.1'
  ./bump_version.sh --dry-run patch
  ./bump_version.sh --no-stage
        """
    )
    parser.add_argument(
        "version",
        nargs="?",
        default="patch",
        help="Bump type ('patch', 'minor', 'major') or explicit version (e.g. '0.3.1'). Default: 'patch'"
    )
    parser.add_argument(
        "--target",
        choices=["lib", "models", "all"],
        default="lib",
        help="Target component to bump: 'lib' (default), 'models', or 'all'"
    )
    parser.add_argument(
        "--lib",
        action="store_true",
        help="Bump library packages only (default)"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Bump OCR model packages only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Bump both library and OCR model packages"
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
    parser.add_argument(
        "--extract-changelog",
        metavar="VERSION",
        help="Extract or generate changelog section for the given version and print to stdout"
    )

    args = parser.parse_args()

    if args.extract_changelog:
        section = extract_changelog_section(args.extract_changelog)
        print(section)
        return

    target = args.target
    if args.models:
        target = "models"
    elif args.all:
        target = "all"
    elif args.lib:
        target = "lib"

    try:
        if target == "models":
            current_ver = get_current_models_version()
            new_ver = parse_new_version(current_ver, args.version)
            print(f"Target          : OCR Models")
            print(f"Current version : {current_ver}")
            print(f"New version     : {new_ver}")
        elif target == "all":
            current_lib = get_current_lib_version()
            new_lib = parse_new_version(current_lib, args.version)
            current_models = get_current_models_version()
            new_models = parse_new_version(current_models, args.version)
            print(f"Target          : Libraries & OCR Models")
            print(f"Current lib ver : {current_lib} -> {new_lib}")
            print(f"Current mod ver : {current_models} -> {new_models}")
            new_ver = new_lib
        else:
            current_ver = get_current_lib_version()
            new_ver = parse_new_version(current_ver, args.version)
            print(f"Target          : Libraries (core engine & bindings)")
            print(f"Current version : {current_ver}")
            print(f"New version     : {new_ver}")
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("Mode            : [DRY-RUN - No files will be modified]")

    if target == "models":
        modified_files = update_models_files(new_ver, current_ver=current_ver, dry_run=args.dry_run)
    elif target == "all":
        modified_files = set()
        modified_files.update(update_lib_files(new_lib, current_ver=current_lib, dry_run=args.dry_run))
        modified_files.update(update_models_files(new_models, current_ver=current_models, dry_run=args.dry_run))
    else:
        modified_files = update_lib_files(new_ver, current_ver=current_ver, dry_run=args.dry_run)

    # Sync Cargo.lock if cargo is installed and library was bumped
    if not args.dry_run and target in ("lib", "all"):
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

    if not args.dry_run and not args.no_stage:
        stage_git_files(modified_files)

    if target == "models":
        print(f"\n✨ OCR models version bump to models-rten-v{new_ver} complete!")
    elif target == "all":
        print(f"\n✨ Full version bump complete (lib: v{new_lib}, models: models-rten-v{new_models})!")
    else:
        print(f"\n✨ Library version bump to v{new_ver} complete!")

    if args.git and not args.dry_run:
        print("\nCreating Git commit and tag(s)...")
        if target == "models":
            commit_msg = f"chore: release models-rten-v{new_ver}"
            tag_name = f"models-rten-v{new_ver}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"], cwd=REPO_ROOT, check=True)
            print(f"  [OK] Created commit: '{commit_msg}'")
            print(f"  [OK] Created git tag: '{tag_name}'")
        elif target == "all":
            commit_msg = f"chore: release v{new_lib} & models-rten-v{new_models}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
            tag_lib = f"v{new_lib}"
            tag_models = f"models-rten-v{new_models}"
            subprocess.run(["git", "tag", "-a", tag_lib, "-m", f"Release {tag_lib}"], cwd=REPO_ROOT, check=True)
            subprocess.run(["git", "tag", "-a", tag_models, "-m", f"Release {tag_models}"], cwd=REPO_ROOT, check=True)
            print(f"  [OK] Created commit: '{commit_msg}'")
            print(f"  [OK] Created git tag: '{tag_lib}'")
            print(f"  [OK] Created git tag: '{tag_models}'")
        else:
            commit_msg = f"chore: release v{new_ver}"
            tag_name = f"v{new_ver}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"], cwd=REPO_ROOT, check=True)
            print(f"  [OK] Created commit: '{commit_msg}'")
            print(f"  [OK] Created git tag: '{tag_name}'")
        print("\nTo push changes and trigger CI/CD release:")
        print(f"  git push origin main --tags")
    elif not args.dry_run:
        print("\nSuggested next steps:")
        if target == "models":
            print(f'  git commit -m "chore: release models-rten-v{new_ver}"')
            print(f'  git tag -a models-rten-v{new_ver} -m "Release models-rten-v{new_ver}"')
        elif target == "all":
            print(f'  git commit -m "chore: release v{new_lib} & models-rten-v{new_models}"')
            print(f'  git tag -a v{new_lib} -m "Release v{new_lib}"')
            print(f'  git tag -a models-rten-v{new_models} -m "Release models-rten-v{new_models}"')
        else:
            print(f'  git commit -m "chore: release v{new_ver}"')
            print(f'  git tag -a v{new_ver} -m "Release v{new_ver}"')
        print(f"  git push origin main --tags")

if __name__ == "__main__":
    main()


