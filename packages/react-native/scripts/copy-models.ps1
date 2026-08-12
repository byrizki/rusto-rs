# PowerShell script to copy model files for bundling with React Native packages
# Run from repository root: .\packages\react-native\scripts\copy-models.ps1

$ErrorActionPreference = "Stop"

param (
    [string]$ModelVersion = "PPOCR_v6"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\..\..\..\"
$ModelsSource = Join-Path $RepoRoot "models\$ModelVersion"
if (-not (Test-Path $ModelsSource)) {
    $ModelsSource = Join-Path $RepoRoot "models\PPOCR_v5"
}
$RNPackage = Join-Path $RepoRoot "packages\react-native"
$AndroidPackage = Join-Path $RepoRoot "packages\android"

Write-Host "Copying model files for React Native bundling..." -ForegroundColor Cyan
Write-Host "Repository root: $RepoRoot"
Write-Host "Model source: $ModelsSource"
Write-Host ""

# Check if source models exist; if not, download on the fly
if (-not (Test-Path "$ModelsSource\det.mnn") -or -not (Test-Path "$ModelsSource\rec.mnn") -or -not (Test-Path "$ModelsSource\dict.txt")) {
    Write-Host "Models not found in $ModelsSource. Downloading default PP-OCRv6 tiny models..." -ForegroundColor Yellow
    & "$RepoRoot\scripts\download_models.ps1" -OutputDir "$ModelsSource"
}

# Android: Copy to main android package assets
Write-Host "📦 Android Setup..." -ForegroundColor Yellow
$AndroidAssets = Join-Path $AndroidPackage "src\main\assets"
New-Item -ItemType Directory -Path $AndroidAssets -Force | Out-Null

Copy-Item -Path "$ModelsSource\det.mnn" -Destination $AndroidAssets -Force
Copy-Item -Path "$ModelsSource\rec.mnn" -Destination $AndroidAssets -Force
Copy-Item -Path "$ModelsSource\dict.txt" -Destination $AndroidAssets -Force

$detSize = (Get-Item "$AndroidAssets\det.mnn").Length / 1MB
$recSize = (Get-Item "$AndroidAssets\rec.mnn").Length / 1MB
$dictSize = (Get-Item "$AndroidAssets\dict.txt").Length / 1KB

Write-Host "✓ Copied models to $AndroidAssets" -ForegroundColor Green
Write-Host "  - det.mnn ($([math]::Round($detSize, 2)) MB)"
Write-Host "  - rec.mnn ($([math]::Round($recSize, 2)) MB)"
Write-Host "  - dict.txt ($([math]::Round($dictSize, 2)) KB)"
Write-Host ""

# iOS: Copy to react-native ios/models directory
Write-Host "🍎 iOS Setup..." -ForegroundColor Yellow
$IOSModels = Join-Path $RNPackage "ios\models"
New-Item -ItemType Directory -Path $IOSModels -Force | Out-Null

Copy-Item -Path "$ModelsSource\det.mnn" -Destination $IOSModels -Force
Copy-Item -Path "$ModelsSource\rec.mnn" -Destination $IOSModels -Force
Copy-Item -Path "$ModelsSource\dict.txt" -Destination $IOSModels -Force

$iosDetSize = (Get-Item "$IOSModels\det.mnn").Length / 1MB
$iosRecSize = (Get-Item "$IOSModels\rec.mnn").Length / 1MB
$iosDictSize = (Get-Item "$IOSModels\dict.txt").Length / 1KB

Write-Host "✓ Copied models to $IOSModels" -ForegroundColor Green
Write-Host "  - det.mnn ($([math]::Round($iosDetSize, 2)) MB)"
Write-Host "  - rec.mnn ($([math]::Round($iosRecSize, 2)) MB)"
Write-Host "  - dict.txt ($([math]::Round($iosDictSize, 2)) KB)"
Write-Host ""

# Calculate total size
$androidTotal = (Get-ChildItem -Path $AndroidAssets -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$iosTotal = (Get-ChildItem -Path $IOSModels -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "✅ Model files copied successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Total bundled size:"
Write-Host "  Android: $([math]::Round($androidTotal, 2)) MB"
Write-Host "  iOS: $([math]::Round($iosTotal, 2)) MB"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Build Android: cd packages\react-native\android && .\gradlew assembleRelease"
Write-Host "  2. Build iOS: cd packages\react-native\ios && pod install"
Write-Host "  3. Use in app: await initialize() // No parameters needed!"
