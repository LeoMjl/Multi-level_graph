[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "no_retrieval",
        "rag_query_to_slice_text",
        "agentrunbook_r_text"
    )]
    [string[]]$Method,
    [ValidateSet("small", "medium")]
    [string]$Tier = "small",
    [ValidateSet("web", "enterprise", "both")]
    [string]$Domain = "both",
    [int]$Limit = 0,
    [int]$ShuffleSeed = -1,
    [switch]$TextOnly,
    [string]$QuestionIdsFile = "",
    [string]$RepositoryRoot = "",
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [string]$MemoryCacheRoot = "",
    [string]$Python = "python",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $projectRoot "data\raw\repos\longmemeval_v2\repository"
}
if (-not $DataRoot) {
    $DataRoot = Join-Path $projectRoot "data\raw\repos\longmemeval_v2\dataset"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot "results\longmemeval_v2_official"
}
if (-not $MemoryCacheRoot) {
    $MemoryCacheRoot = Join-Path $projectRoot `
        "data\cache\longmemeval_v2_official\domain_memory_v1"
}

$runner = Join-Path $RepositoryRoot "evaluation\run_eval.py"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Official LongMemEval-V2 runner is missing: $runner"
}
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "questions.jsonl"))) {
    throw "Official LongMemEval-V2 data is missing: $DataRoot"
}
$reservedMemoryArgs = @(
    $ExtraArgs | Where-Object {
        $_ -eq "--save-memory" -or
        $_ -eq "--load-memory-dir" -or
        $_ -eq "--memory-cache-dir"
    }
)
if ($reservedMemoryArgs.Count -gt 0) {
    throw "Memory persistence is owned by -MemoryCacheRoot; remove: $($reservedMemoryArgs -join ', ')"
}

$domains = if ($Domain -eq "both") { @("web", "enterprise") } else { @($Domain) }
$releasedQuestions = @()
if ($TextOnly -or $QuestionIdsFile) {
    $releasedQuestions = @(
        Get-Content -LiteralPath (Join-Path $DataRoot "questions.jsonl") |
            Where-Object { $_.Trim() } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
}
$requestedQuestionIds = @()
if ($QuestionIdsFile) {
    $resolvedIdsFile = (Resolve-Path -LiteralPath $QuestionIdsFile).Path
    if ([System.IO.Path]::GetExtension($resolvedIdsFile) -eq ".json") {
        $manifest = Get-Content -LiteralPath $resolvedIdsFile -Raw | ConvertFrom-Json
        if ($null -ne $manifest.samples) {
            $requestedQuestionIds = @(
                $manifest.samples | ForEach-Object { [string]$_.episode_id }
            )
        } else {
            $requestedQuestionIds = @(
                $manifest | ForEach-Object {
                    if ($_ -is [string]) { [string]$_ }
                    elseif ($null -ne $_.episode_id) { [string]$_.episode_id }
                    else { [string]$_.id }
                }
            )
        }
    } else {
        $requestedQuestionIds = @(
            Get-Content -LiteralPath $resolvedIdsFile |
                ForEach-Object { $_.Trim() } |
                Where-Object { $_ }
        )
    }
    $requestedQuestionIds = @($requestedQuestionIds | Where-Object { $_ } | Select-Object -Unique)
    if ($requestedQuestionIds.Count -eq 0) {
        throw "Question id file contains no ids: $resolvedIdsFile"
    }
    $knownIds = @($releasedQuestions | ForEach-Object { [string]$_.id })
    $missingIds = @($requestedQuestionIds | Where-Object { $_ -notin $knownIds })
    if ($missingIds.Count -gt 0) {
        throw "Unknown LongMemEval-V2 question ids: $($missingIds -join ', ')"
    }
}
foreach ($methodName in $Method) {
    foreach ($domainName in $domains) {
        $outputDir = Join-Path $OutputRoot "${methodName}_${domainName}_${Tier}"
        $completedMetrics = Join-Path $outputDir "aggregated_metrics.json"
        $memoryWorkspace = Join-Path $outputDir "memory_workspace"
        if ((Test-Path -LiteralPath $memoryWorkspace) -and
            -not (Test-Path -LiteralPath $completedMetrics)) {
            $resolvedOutputDir = [IO.Path]::GetFullPath($outputDir).TrimEnd(
                [IO.Path]::DirectorySeparatorChar
            )
            $resolvedWorkspace = [IO.Path]::GetFullPath($memoryWorkspace)
            $expectedPrefix = $resolvedOutputDir + [IO.Path]::DirectorySeparatorChar
            if (-not $resolvedWorkspace.StartsWith(
                $expectedPrefix,
                [StringComparison]::OrdinalIgnoreCase
            )) {
                throw "Refusing out-of-result workspace cleanup: $resolvedWorkspace"
            }
            Remove-Item -LiteralPath $resolvedWorkspace -Recurse -Force
            Write-Output "Removed incomplete memory workspace: $resolvedWorkspace"
        }
        $effectiveExtraArgs = @($ExtraArgs)
        $arguments = @(
            $runner,
            "--method", $methodName,
            "--data-root", $DataRoot,
            "--domain", $domainName,
            "--tier", $Tier,
            "--output-dir", $outputDir
        )
        if ($methodName -ne "no_retrieval") {
            $memoryCacheDir = Join-Path $MemoryCacheRoot `
                "${methodName}_${domainName}_${Tier}"
            $arguments += @("--memory-cache-dir", $memoryCacheDir)
            Write-Output "Domain memory cache: $memoryCacheDir"
        }
        if ($Limit -gt 0) {
            $arguments += @("--limit", [string]$Limit)
        }
        if ($ShuffleSeed -ge 0) {
            $arguments += @("--shuffle-questions-seed", [string]$ShuffleSeed)
        }
        if ($TextOnly -or $QuestionIdsFile) {
            $domainQuestionIds = @(
                $releasedQuestions |
                    Where-Object {
                        $_.domain -eq $domainName -and
                        (-not $TextOnly -or -not $_.image) -and
                        (-not $QuestionIdsFile -or [string]$_.id -in $requestedQuestionIds)
                    } |
                    ForEach-Object { [string]$_.id }
            )
            if ($domainQuestionIds.Count -eq 0) {
                throw "No selected questions for domain=$domainName"
            }
            $arguments += "--question-ids"
            $arguments += $domainQuestionIds
        }
        $arguments += $effectiveExtraArgs
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Official LongMemEval-V2 run failed: method=$methodName domain=$domainName tier=$Tier"
        }
    }
}
