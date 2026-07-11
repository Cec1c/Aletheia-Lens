[CmdletBinding()]
param(
    [string]$Tag = "models-v1",
    [string]$Repository = "Cec1c/Aletheia-Lens",
    [switch]$Replace,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repositoryRoot ".github\model-assets.sha256"
$modelsRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot "models")) + [IO.Path]::DirectorySeparatorChar
$allowedModelPaths = @(
    "models/mrcnn/weights.onnx",
    "models/esrgan/4x-Fatal-Pixels.onnx",
    "models/esrgan/4x-Fatal-Pixels.onnx.data"
)

$records = foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }
    if ($line -notmatch '^([0-9A-Fa-f]{64})\s+\*(.+)$') {
        throw "Invalid model manifest line: $line"
    }

    $relativePath = $matches[2].Trim().Replace('\', '/')
    if ($relativePath -notin $allowedModelPaths) {
        throw "Model manifest path is not allowed: $relativePath"
    }
    $resolvedPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $relativePath))
    if (-not $resolvedPath.StartsWith($modelsRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Model manifest path escapes the models directory: $relativePath"
    }

    [PSCustomObject]@{
        Hash = $matches[1].ToUpperInvariant()
        RelativePath = $relativePath
        Path = $resolvedPath
    }
}

foreach ($allowedPath in $allowedModelPaths) {
    if ($allowedPath -notin $records.RelativePath) {
        throw "Model manifest is missing required path: $allowedPath"
    }
}

foreach ($record in $records) {
    if (-not (Test-Path -LiteralPath $record.Path)) {
        throw "Missing model asset: $($record.RelativePath)"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $record.Path).Hash
    if ($actualHash -ne $record.Hash) {
        throw "Hash mismatch for $($record.RelativePath): expected $($record.Hash), got $actualHash"
    }
}

if ($VerifyOnly) {
    Write-Output "All model assets match .github/model-assets.sha256."
    return
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) was not found."
}

& gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run gh auth login first."
}

$nativeErrorPreference = $PSNativeCommandUseErrorActionPreference
$PSNativeCommandUseErrorActionPreference = $false
try {
    & gh release view $Tag --repo $Repository --json tagName 2>$null | Out-Null
    $releaseExists = $LASTEXITCODE -eq 0
}
finally {
    $PSNativeCommandUseErrorActionPreference = $nativeErrorPreference
}

$uploadAssets = @($records | ForEach-Object { $_.Path }) + @($manifestPath)
if ($releaseExists) {
    if (-not $Replace) {
        throw "Release '$Tag' already exists. Re-run with -Replace to update its assets."
    }
    & gh release upload $Tag @uploadAssets --clobber --repo $Repository
}
else {
    & gh release create $Tag @uploadAssets `
        --repo $Repository `
        --title "Aletheia-Lens ONNX models" `
        --notes "Pinned ONNX runtime assets used by GitHub Actions builds." `
        --latest=false
}

if ($LASTEXITCODE -ne 0) {
    throw "Publishing model release '$Tag' failed with exit code $LASTEXITCODE."
}

$published = & gh release view $Tag --repo $Repository --json assets | ConvertFrom-Json
foreach ($assetPath in $uploadAssets) {
    $localAsset = Get-Item -LiteralPath $assetPath
    $remoteAsset = $published.assets | Where-Object name -eq $localAsset.Name | Select-Object -First 1
    if (-not $remoteAsset -or $remoteAsset.size -ne $localAsset.Length) {
        throw "Published asset verification failed for $($localAsset.Name)."
    }
    if ($remoteAsset.digest) {
        $localDigest = "sha256:$((Get-FileHash -Algorithm SHA256 -LiteralPath $localAsset.FullName).Hash.ToLowerInvariant())"
        if ($remoteAsset.digest.ToLowerInvariant() -ne $localDigest) {
            throw "Published asset digest mismatch for $($localAsset.Name)."
        }
    }
}

Write-Output "Model release '$Tag' in '$Repository' is ready and uploaded assets were verified."
