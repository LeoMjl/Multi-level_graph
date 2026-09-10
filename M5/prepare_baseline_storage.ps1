param(
    [string]$ArtifactRoot = (Join-Path $PSScriptRoot 'artifacts')
)
$ErrorActionPreference = 'Stop'
$artifact = [IO.Path]::GetFullPath($ArtifactRoot)
if ($artifact.TrimEnd('\') -eq [IO.Path]::GetPathRoot($artifact).TrimEnd('\')) {
    throw 'ArtifactRoot cannot be a drive root'
}
$accounts = @('CodexSandboxOffline', 'CodexSandboxOnline') | ForEach-Object {
    (Get-LocalUser -Name $_).SID
}
foreach ($name in @('runs', 'validation', 'cache', 'actors')) {
    $directory = Join-Path $artifact $name
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    if ($name -eq 'actors') { continue }
    $acl = Get-Acl -LiteralPath $directory
    foreach ($sid in $accounts) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            $sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Deny'
        )
        $acl.SetAccessRule($rule)
    }
    Set-Acl -LiteralPath $directory -AclObject $acl
}
& (Join-Path $PSScriptRoot 'methods\TaskGraph\set_isolation_acl.ps1') -Action Apply
if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw 'M5 ACL setup failed' }
[pscustomobject]@{ artifact_root=$artifact; protected=@('runs','validation','cache'); actors='actors' } |
    ConvertTo-Json -Compress
