param(
    [ValidateSet("Apply", "Remove", "Verify")]
    [string]$Action = "Apply",
    [string]$RepositoryRoot = (Join-Path $PSScriptRoot "..\..")
)

$ErrorActionPreference = "Stop"
$expected = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$target = (Resolve-Path -LiteralPath $RepositoryRoot).Path
if (-not [string]::Equals($expected, $target, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing ACL mutation outside the M5 repository: $target"
}

$accounts = foreach ($name in @("CodexSandboxOffline", "CodexSandboxOnline")) {
    $account = Get-LocalUser -Name $name -ErrorAction SilentlyContinue
    if ($null -eq $account) {
        throw "Required Windows Codex sandbox account is missing: $name"
    }
    [pscustomobject]@{ Name = $account.Name; Sid = $account.SID.Value }
}

# Protect the public M5 experiment, including prompts and each method snapshot.
$protectedRoots = @($target)
$probes = @(
    (Join-Path $target "hooks_gold.jsonl"),
    (Join-Path $target "hook_schedule.jsonl"),
    (Join-Path $target "story_bible.md"),
    (Join-Path $target "prompts\chapters_281_320.jsonl"),
    (Join-Path $PSScriptRoot "src\mlg\m5\dataset.py")
)

if ($Action -ne "Verify") {
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    foreach ($root in $protectedRoots) {
        foreach ($account in $accounts) {
            if ($Action -eq "Apply") {
                & icacls.exe $root /deny "*$($account.Sid):(OI)(CI)(F)" /T /C /Q 2>$null | Out-Null
            } else {
                & icacls.exe $root /remove:d "*$($account.Sid)" /T /C /Q 2>$null | Out-Null
            }
        }
    }
    $ErrorActionPreference = $savedErrorPreference
}

function Get-MissingDeny([string]$Path) {
    $denySids = foreach ($rule in (Get-Acl -LiteralPath $Path).Access) {
        if ($rule.AccessControlType -ne "Deny") { continue }
        try {
            $rule.IdentityReference.Translate(
                [System.Security.Principal.SecurityIdentifier]
            ).Value
        } catch { continue }
    }
    @($accounts.Sid | Where-Object { $denySids -notcontains $_ })
}

$checks = foreach ($path in @($protectedRoots + $probes)) {
    $missing = @(Get-MissingDeny $path)
    [pscustomobject]@{
        path = $path
        deny_acl_present = ($missing.Count -eq 0)
        missing_sids = $missing
    }
}
$bad = @($checks | Where-Object {
    if ($Action -eq "Remove") { $_.deny_acl_present } else { -not $_.deny_acl_present }
})
if ($bad.Count -gt 0) {
    throw "M5 ACL verification failed for: $($bad.path -join ', ')"
}

[pscustomobject]@{
    schema = "m5-codex-sandbox-acl-v2"
    action = $Action
    repository_root = $target
    protected_roots = $protectedRoots
    sandbox_accounts = $accounts
    checks = $checks
} | ConvertTo-Json -Depth 5
