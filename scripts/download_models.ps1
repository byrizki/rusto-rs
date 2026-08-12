# PowerShell script to download default bundled OCR models on the fly
# Priority:
# 1. https://github.com/byrizki/rusto-rs-models/releases
# 2. https://www.modelscope.cn/models/RapidAI/RapidOCR

param (
    [string]$OutputDir = "",
    [string]$Tier = "tiny",
    [string]$Version = "v1.0.0"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\..\"

if ([string]::IsNullOrEmpty($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "models\PPOCR_v6"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

Write-Host "=== Downloading PP-OCRv6 ($Tier tier) models ===" -ForegroundColor Cyan
Write-Host "Destination: $OutputDir"

$GhReleaseUrl = "https://github.com/byrizki/rusto-rs-models/releases/download/$Version"
$GhLatestUrl = "https://github.com/byrizki/rusto-rs-models/releases/latest/download"
$ModelScopeBase = "https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath="

function Download-File {
    param (
        [string]$Filename,
        [string]$GhName,
        [string]$MsPath
    )

    $Dest = Join-Path $OutputDir $Filename

    if (Test-Path $Dest) {
        $fileSize = (Get-Item $Dest).Length / 1MB
        Write-Host "✓ Already exists: $Filename ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
        return
    }

    Write-Host "Downloading $Filename..." -ForegroundColor Yellow

    # 1. Try specific GitHub release
    try {
        Invoke-WebRequest -Uri "$GhReleaseUrl/$GhName" -OutFile $Dest -ErrorAction Stop
        if ((Get-Item $Dest).Length -gt 0) {
            Write-Host "✓ Downloaded from GitHub release ($Version): $Filename" -ForegroundColor Green
            return
        }
    } catch {}

    # 2. Try latest GitHub release
    try {
        Invoke-WebRequest -Uri "$GhLatestUrl/$GhName" -OutFile $Dest -ErrorAction Stop
        if ((Get-Item $Dest).Length -gt 0) {
            Write-Host "✓ Downloaded from GitHub release (latest): $Filename" -ForegroundColor Green
            return
        }
    } catch {}

    # 3. Fallback to ModelScope
    Write-Host "  Downloading from ModelScope fallback..."
    try {
        Invoke-WebRequest -Uri "$ModelScopeBase$MsPath" -OutFile $Dest -ErrorAction Stop
        Write-Host "✓ Downloaded from ModelScope: $Filename" -ForegroundColor Green
        return
    } catch {
        Write-Host "❌ Failed to download $Filename : $_" -ForegroundColor Red
        throw
    }
}

Download-File -Filename "det.mnn" -GhName "ppocrv6_det_${Tier}.mnn" -MsPath "mnn%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_${Tier}.mnn"
Download-File -Filename "rec.mnn" -GhName "ppocrv6_rec_${Tier}.mnn" -MsPath "mnn%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_${Tier}.mnn"

if ($Tier -eq "tiny") {
    Download-File -Filename "dict.txt" -GhName "ppocrv6_tiny_dict.txt" -MsPath "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny%2Fppocrv6_tiny_dict.txt"
} else {
    Download-File -Filename "dict.txt" -GhName "ppocrv6_dict.txt" -MsPath "paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_small%2Fppocrv6_dict.txt"
}

Write-Host "`n✅ Models ready in $OutputDir" -ForegroundColor Green
