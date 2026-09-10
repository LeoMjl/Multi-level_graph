param(
    [ValidateSet("Apply", "Verify")]
    [string]$Action = "Verify",
    [string]$HistoryRoot = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else {
        Join-Path $env:USERPROFILE ".codex"
    })
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $HistoryRoot).Path.TrimEnd('\')
$expected = if ($env:CODEX_HOME) { $env:CODEX_HOME } else {
    Join-Path $env:USERPROFILE ".codex"
}
$expected = (Resolve-Path -LiteralPath $expected).Path.TrimEnd('\')
if (-not [string]::Equals($root, $expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw "HistoryRoot must be the current Codex data directory"
}
$accounts = @("CodexSandboxOffline", "CodexSandboxOnline") | ForEach-Object {
    Get-LocalUser -Name $_ -ErrorAction Stop
}
# Protect atomically replaced root-level metadata. The root and descendant
# directories remain unaffected; only direct child files inherit ReadData deny.
$inheritance = [Security.AccessControl.InheritanceFlags]::ObjectInherit
$propagation = [Security.AccessControl.PropagationFlags]::InheritOnly -bor
    [Security.AccessControl.PropagationFlags]::NoPropagateInherit
$directFileRules = @($accounts | ForEach-Object {
    [Security.AccessControl.FileSystemAccessRule]::new(
        $_.SID, [Security.AccessControl.FileSystemRights]::ReadData,
        $inheritance, $propagation, [Security.AccessControl.AccessControlType]::Deny
    )
})
if ($Action -eq "Apply") {
    $rootAcl = Get-Acl -LiteralPath $root
    foreach ($rule in $directFileRules) { $rootAcl.AddAccessRule($rule) }
    Set-Acl -LiteralPath $root -AclObject $rootAcl
}
$rootDenies = @((Get-Acl -LiteralPath $root).Access | Where-Object {
    $_.AccessControlType -eq "Deny" -and
    $_.FileSystemRights -eq [Security.AccessControl.FileSystemRights]::ReadData -and
    $_.InheritanceFlags -eq $inheritance -and $_.PropagationFlags -eq $propagation
} | ForEach-Object {
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
})
if (@($accounts | Where-Object { $rootDenies -notcontains $_.SID.Value }).Count) {
    throw "Codex direct-child-file inheritance guard is missing"
}
$directoryNames = @(
    "sessions", "archived_sessions", "memories", "attachments", "automations",
    "log_backups", "sqlite", "process_manager", "visualizations",
    "dictation-history", "ambient-suggestions", "browser\sessions"
)
$targets = [Collections.Generic.List[string]]::new()
foreach ($name in $directoryNames) {
    $path = Join-Path $root $name
    if (Test-Path -LiteralPath $path -PathType Container) { $targets.Add($path) }
}
# These patterns add explicit denies to historical data only. Runtime bins
# retain their ACLs; direct auth/config files inherit only the guard above.
foreach ($pattern in @(
    "*.sqlite*", "history.jsonl*", "session_index.jsonl*",
    "transcription-history.jsonl*", "sandbox*.log"
)) {
    Get-ChildItem -LiteralPath $root -Filter $pattern -File -Force |
        ForEach-Object { $targets.Add($_.FullName) }
}
$sandboxLogs = Join-Path $root ".sandbox"
if (Test-Path -LiteralPath $sandboxLogs -PathType Container) {
    Get-ChildItem -LiteralPath $sandboxLogs -Filter "*.log" -File |
        ForEach-Object { $targets.Add($_.FullName) }
}
$paths = @($targets | Sort-Object -Unique)
foreach ($path in $paths) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing an ACL target outside the Codex history directory: $resolved"
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing to modify a reparse-point ACL: $resolved"
    }
    if ($Action -eq "Apply") {
        foreach ($account in $accounts) {
            $grant = if ($item.PSIsContainer) { "(OI)(CI)(F)" } else { "(F)" }
            $arguments = @($resolved, "/deny", "*$($account.SID.Value):$grant", "/C", "/Q", "/L")
            if ($item.PSIsContainer) { $arguments += "/T" }
            & icacls.exe @arguments | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Failed to protect history path: $resolved" }
        }
    }
}
$failed = [Collections.Generic.List[string]]::new()
foreach ($path in $paths) {
    $denied = @((Get-Acl -LiteralPath $path).Access | Where-Object {
        $_.AccessControlType -eq "Deny" -and
        ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -eq
        [Security.AccessControl.FileSystemRights]::FullControl
    } | ForEach-Object {
        $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    })
    foreach ($account in $accounts) {
        if ($denied -notcontains $account.SID.Value) { $failed.Add($path) }
    }
}
if ($failed.Count) { throw "History ACL verification failed: $($failed -join ', ')" }
[ordered]@{
    schema = "m5-codex-history-acl-v1"
    action = $Action
    history_root = $root
    protected_target_count = $paths.Count
    direct_child_files_read_denied = $true
    sandbox_accounts = @($accounts.Name)
    verified = $true
} | ConvertTo-Json -Depth 3
