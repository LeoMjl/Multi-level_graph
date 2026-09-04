[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$OfficialRoot = "",
    [string]$ToolRoot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$expectedCommit = "aa4ed9f4737ad98bd706663f01d63623c3427812"
if (-not $OfficialRoot) {
    $OfficialRoot = Join-Path $projectRoot "data\stabletoolbench"
}
if (-not $ToolRoot) {
    $ToolRoot = Join-Path $projectRoot "data\official_query_tools"
}

if (-not (Test-Path -LiteralPath (Join-Path $OfficialRoot ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OfficialRoot) | Out-Null
    & git clone https://github.com/THUNLP-MT/StableToolBench.git $OfficialRoot
    if ($LASTEXITCODE -ne 0) { throw "StableToolBench clone failed" }
}

& git -C $OfficialRoot fetch origin $expectedCommit --depth 1
if ($LASTEXITCODE -ne 0) { throw "StableToolBench fetch failed" }
& git -C $OfficialRoot checkout --detach $expectedCommit
if ($LASTEXITCODE -ne 0) { throw "StableToolBench checkout failed" }
$actualCommit = (& git -C $OfficialRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $expectedCommit) {
    throw "Unexpected StableToolBench source revision"
}

& $Python (Join-Path $PSScriptRoot "build_tool_environment.py") `
    --query-root (Join-Path $OfficialRoot "solvable_queries") `
    --output $ToolRoot
if ($LASTEXITCODE -ne 0) { throw "Tool environment build failed" }

Write-Output "StableToolBench checkout: $OfficialRoot"
Write-Output "Official-query tools: $ToolRoot"
