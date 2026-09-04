[CmdletBinding()]
param(
    [string]$RepositoryRoot = "",
    [string]$DataRoot = "",
    [ValidateSet("small", "medium")]
    [string]$Tier = "small",
    [string]$Python = "python",
    [switch]$SkipScreenshots
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $projectRoot "data\raw\repos\longmemeval_v2\repository"
}
if (-not $DataRoot) {
    $DataRoot = Join-Path $projectRoot "data\raw\repos\longmemeval_v2\dataset"
}

if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot "data\download_data.py"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RepositoryRoot) | Out-Null
    & git clone --depth 1 https://github.com/xiaowu0162/LongMemEval-V2.git $RepositoryRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone LongMemEval-V2." }
}

& $Python (Join-Path $RepositoryRoot "data\download_data.py") --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "LongMemEval-V2 download failed." }

if (-not $SkipScreenshots) {
    & $Python (Join-Path $RepositoryRoot "data\prepare_data.py") --data-root $DataRoot --mode symlink
    if ($LASTEXITCODE -ne 0) { throw "LongMemEval-V2 screenshot preparation failed." }
}

$validationArgs = @(
    (Join-Path $RepositoryRoot "data\validate_data.py"),
    "--data-root", $DataRoot,
    "--tier", $Tier
)
if ($SkipScreenshots) {
    $validationArgs += "--no-check-screenshots"
}
& $Python @validationArgs
if ($LASTEXITCODE -ne 0) { throw "LongMemEval-V2 validation failed." }

Write-Host "LongMemEval-V2 is ready at $DataRoot"
