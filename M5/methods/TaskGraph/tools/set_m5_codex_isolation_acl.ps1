param(
    [ValidateSet("Apply", "Remove", "Verify")]
    [string]$Action = "Apply"
)

$ErrorActionPreference = "Stop"
$script = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\set_isolation_acl.ps1")).Path
& $script -Action $Action
exit $LASTEXITCODE
