#!/usr/bin/env bash
# Stage release-native artifacts after react-native-rusto.tgz installation.
# Never publish staged binaries in npm; CI injects them into a disposable example.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  patch-rn-example-artifacts.sh --platform android|ios --app DIR --core FILE --model FILE --version VERSION [--model-pod NAME]
USAGE
  exit 2
}
platform= app= core= model= version= model_pod=
while (($#)); do
  case "$1" in
    --platform) shift; (($#)) || usage; platform=$1; shift ;;
    --app) shift; (($#)) || usage; app=$1; shift ;;
    --core) shift; (($#)) || usage; core=$1; shift ;;
    --model) shift; (($#)) || usage; model=$1; shift ;;
    --version) shift; (($#)) || usage; version=$1; shift ;;
    --model-pod) shift; (($#)) || usage; model_pod=$1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$platform" && -n "$app" && -n "$core" && -n "$model" && -n "$version" ]] || usage
[[ -d "$app/node_modules/react-native-rusto" ]] || { echo "react-native-rusto is not installed: $app" >&2; exit 1; }
[[ -s "$core" && -s "$model" ]] || { echo "missing staged release artifact" >&2; exit 1; }

case "$platform" in
  android)
    [[ "$core" == *.aar && "$model" == *.aar ]] || { echo "Android requires .aar artifacts" >&2; exit 1; }
    libs="$app/node_modules/react-native-rusto/android/libs"
    rm -rf "$libs"
    mkdir -p "$libs"
    cp "$core" "$model" "$libs/"
    test "$(find "$libs" -maxdepth 1 -name '*.aar' | wc -l | tr -d ' ')" = 2
    # Artifact mode must win before bridge's JitPack fallback.
    grep -Fq 'else if (hasLocalAar)' "$app/node_modules/react-native-rusto/android/build.gradle"
    ! grep -Fq "include ':rusto-android'" "$app/android/settings.gradle"
    ;;
  ios)
    [[ "$core" == *.zip && "$model" == *.zip && -n "$model_pod" ]] || {
      echo "iOS requires core/model .zip plus --model-pod" >&2; exit 1;
    }
    core_pod="${RUSTO_IOS_CORE_POD_DIR:-packages/ios}"
    test -f "$core_pod/RustO.podspec"
    rm -rf "$core_pod/RustO.xcframework"
    unzip -q "$core" 'RustO.xcframework/*' -d "$core_pod"
    test -d "$core_pod/RustO.xcframework"

    model_dir="$app/ios/.rusto-artifacts/$model_pod"
    rm -rf "$model_dir"
    mkdir -p "$model_dir"
    model_dir="$(cd "$model_dir" && pwd)"
    unzip -q "$model" -d "$model_dir"
    cp "$core_pod/models/$model_pod.podspec" "$model_dir/$model_pod.podspec"
    python3 - "$version" "$core_pod/RustO.podspec" "$model_dir/$model_pod.podspec" <<'PY'
import re, sys
version, *paths = sys.argv[1:]
for path in paths:
    text = open(path).read()
    changed = re.sub(r"s\.version\s*=\s*['\"][^'\"]+['\"]", f"s.version          = '{version}'", text, count=1)
    if changed == text:
        raise SystemExit(f"no version declaration in {path}")
    open(path, 'w').write(changed)
PY
    podfile="$app/ios/Podfile"
    marker="# RustO CI artifact model override"
    python3 - "$podfile" "$marker" "$model_pod" "$model_dir" <<'PY'
import sys
path, marker, pod, directory = sys.argv[1:]
text = open(path).read()
line = f"  pod '{pod}', :path => '{directory}'"
text = '\n'.join(x for x in text.split('\n') if marker not in x and not x.strip().startswith(f"pod '{pod}',"))
needle = "  # Stage RustO.xcframework from Build & Release before pod install.\n"
if needle not in text:
    raise SystemExit('missing RustO local-pod marker in Podfile')
text = text.replace(needle, needle + f"  {marker}\n" + line + "\n", 1)
open(path, 'w').write(text)
PY
    grep -Fq "pod '$model_pod', :path => '$model_dir'" "$podfile"
    ;;
  *) usage ;;
esac
