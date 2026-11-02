#!/usr/bin/env bash
# Download binding artifacts from a Build & Release run. Supports both current
# split artifacts and pre-split aggregate artifacts so manual validation can
# inspect historical release runs.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: download-binding-artifacts.sh --run-id ID --platform android|ios|dotnet --output DIR
Requires GH_TOKEN and GitHub CLI authentication.
USAGE
  exit 2
}
run_id= platform= output=
while (($#)); do
  case "$1" in
    --run-id) shift; (($#)) || usage; run_id=$1; shift ;;
    --platform) shift; (($#)) || usage; platform=$1; shift ;;
    --output) shift; (($#)) || usage; output=$1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$run_id" && -n "$platform" && -n "$output" ]] || usage
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

names="$(gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/artifacts?per_page=100" --jq '.artifacts[] | select(.expired | not) | .name')"
has() { grep -Fxq "$1" <<<"$names"; }
download() { gh run download "$run_id" --name "$1" --dir "$2"; }
mkdir -p "$output"

case "$platform" in
  android)
    if has rusto-android-core-aar; then
      download rusto-android-core-aar "$output/rusto-android-core-aar"
      if has rusto-android-model-aars; then
        download rusto-android-model-aars "$output/rusto-android-model-aars"
      fi
    elif has rusto-android-aar; then
      echo "::warning::Build run uses legacy aggregate rusto-android-aar; normalizing it to split layout."
      legacy="$output/.legacy-android"
      download rusto-android-aar "$legacy"
      core="$(find "$legacy" -type f -name 'rusto-android-*.aar' ! -name 'rusto-models-*.aar' -print -quit)"
      [[ -n "$core" ]] || { echo "legacy Android artifact lacks required core AAR" >&2; exit 1; }
      mkdir -p "$output/rusto-android-core-aar"
      cp "$core" "$output/rusto-android-core-aar/"
      model="$(find "$legacy" -type f -name 'rusto-models-ppocrv6-tiny-*.aar' -print -quit)"
      if [[ -n "$model" ]]; then
        mkdir -p "$output/rusto-android-model-aars/ppocrv6-tiny"
        cp "$model" "$output/rusto-android-model-aars/ppocrv6-tiny/"
      fi
    else
      echo "Build run $run_id has neither split nor legacy Android artifact." >&2; exit 1
    fi
    ;;
  ios)
    if has rusto-ios-core-xcframework; then
      download rusto-ios-core-xcframework "$output/rusto-ios-core-xcframework"
      if has rusto-ios-model-resources; then
        download rusto-ios-model-resources "$output/rusto-ios-model-resources"
      fi
    elif has rusto-ios-xcframework; then
      echo "::warning::Build run uses legacy aggregate rusto-ios-xcframework; normalizing it to split layout."
      legacy="$output/.legacy-ios"
      download rusto-ios-xcframework "$legacy"
      core="$(find "$legacy" -type f -name 'RustO.xcframework.zip' -print -quit)"
      [[ -n "$core" ]] || { echo "legacy iOS artifact lacks required core bundle" >&2; exit 1; }
      mkdir -p "$output/rusto-ios-core-xcframework"
      cp "$core" "$output/rusto-ios-core-xcframework/"
      model="$(find "$legacy" -type f -name 'RustO-Models-PPOCRv6-Tiny.zip' -print -quit)"
      if [[ -n "$model" ]]; then
        mkdir -p "$output/rusto-ios-model-resources/ppocrv6-tiny"
        cp "$model" "$output/rusto-ios-model-resources/ppocrv6-tiny/"
      fi
    else
      echo "Build run $run_id has neither split nor legacy iOS artifact." >&2; exit 1
    fi
    ;;
  dotnet)
    if has rusto-dotnet-core-nupkg; then
      download rusto-dotnet-core-nupkg "$output/rusto-dotnet-core-nupkg"
      if has rusto-dotnet-model-nupkgs; then
        download rusto-dotnet-model-nupkgs "$output/rusto-dotnet-model-nupkgs"
      fi
    elif has rusto-dotnet-nupkg; then
      echo "::warning::Build run uses legacy aggregate rusto-dotnet-nupkg; normalizing it to split layout."
      legacy="$output/.legacy-dotnet"
      download rusto-dotnet-nupkg "$legacy"
      core="$(find "$legacy" -type f -name 'RustODotnet.[0-9]*.nupkg' -print -quit)"
      [[ -n "$core" ]] || { echo "legacy .NET artifact lacks required core package" >&2; exit 1; }
      mkdir -p "$output/rusto-dotnet-core-nupkg"
      cp "$core" "$output/rusto-dotnet-core-nupkg/"
      model="$(find "$legacy" -type f -name 'RustODotnet.Models.PPOCRv6.Tiny.*.nupkg' -print -quit)"
      if [[ -n "$model" ]]; then
        mkdir -p "$output/rusto-dotnet-model-nupkgs/ppocrv6-tiny"
        cp "$model" "$output/rusto-dotnet-model-nupkgs/ppocrv6-tiny/"
      fi
    else
      echo "Build run $run_id has neither split nor legacy .NET artifact." >&2; exit 1
    fi
    ;;
  *) usage ;;
esac
