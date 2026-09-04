param(
    [ValidateSet("ours", "cot", "dfs")]
    [string]$Method = "ours",
    [ValidateSet(
        "all",
        "G2_instruction",
        "G2_category",
        "G3_instruction"
    )]
    [string]$Subset = "all",
    [string]$OfficialRoot = "",
    [string]$OutputRoot = "",
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$EvaluatorApiPool,
    [string]$EvaluatorModel = "deepseek-v4-flash",
    [ValidateSet("disabled")]
    [string]$EvaluatorThinkingMode = "disabled",
    [double]$EvaluatorTemperature = 0.0,
    [string]$FacModelPath = ""
)

$ErrorActionPreference = "Stop"
$officialCommit = "aa4ed9f4737ad98bd706663f01d63623c3427812"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OfficialRoot) {
    $OfficialRoot = Join-Path $projectRoot "data\raw\repos\stabletoolbench\repository"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot "results\taskgraph"
}
$actualCommit = (git -C $OfficialRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $officialCommit) {
    throw "StableToolBench checkout must be pinned to $officialCommit"
}
if (-not (Test-Path -LiteralPath $EvaluatorApiPool)) {
    throw "Missing StableToolEval API pool: $EvaluatorApiPool"
}

$allSubsets = @(
    "G2_instruction",
    "G2_category",
    "G3_instruction"
)
$subsets = if ($Subset -eq "all") { $allSubsets } else { @($Subset) }
$labels = @{
    ours = "ours_progressive"
    cot = "official_stabletoolbench_cot"
    dfs = "official_stabletoolbench_dfs"
}
$label = $labels[$Method]
$reference = "official_stabletoolbench_cot"
$convertedBase = Join-Path $OutputRoot "converted"
$passBase = Join-Path $OutputRoot "pass_rate"
$passMethod = Join-Path $passBase $label
$preferenceRoot = Join-Path $OutputRoot "preference"
$facBase = Join-Path $OutputRoot "fac"
$facMethod = Join-Path $facBase $label
New-Item -ItemType Directory -Force -Path (
    $passMethod, $preferenceRoot, $facMethod
) | Out-Null

$priorPythonPath = $env:PYTHONPATH
$priorApiPool = $env:API_POOL_FILE
$priorEvalModel = $env:EVAL_MODEL
$priorEvalThinkingMode = $env:EVAL_THINKING_MODE
$priorEvalTemperature = $env:EVAL_TEMPERATURE
$env:PYTHONPATH = "$projectRoot\src;$OfficialRoot;$OfficialRoot\toolbench\inference"
$env:API_POOL_FILE = $EvaluatorApiPool
$env:EVAL_MODEL = $EvaluatorModel
$env:EVAL_THINKING_MODE = $EvaluatorThinkingMode
$env:EVAL_TEMPERATURE = $EvaluatorTemperature.ToString(
    [Globalization.CultureInfo]::InvariantCulture
)
try {
    Push-Location "$OfficialRoot\toolbench\tooleval"
    try {
        & $Python "eval_pass_rate.py" --help *> $null
        $passHelpCode = $LASTEXITCODE
        & $Python "eval_preference.py" --help *> $null
        $preferenceHelpCode = $LASTEXITCODE
    }
    finally { Pop-Location }
    if ($passHelpCode -ne 0 -or $preferenceHelpCode -ne 0) {
        throw "Install the pinned StableToolBench requirements before formal M3 evaluation"
    }
    if ($Method -eq "ours" -and $Subset -eq "all") {
        & $Python "$PSScriptRoot\validate_artifacts.py" `
            --official-root $OfficialRoot `
            --output-root $OutputRoot `
            --method $label `
            --output "$OutputRoot\$label`_artifact_audit.json"
        if ($LASTEXITCODE -ne 0) { throw "Progressive v3 M3 artifact validation failed" }
    }
    foreach ($group in $subsets) {
        $convertedFile = Join-Path $convertedBase "$label\$group.json"
        if (-not (Test-Path -LiteralPath $convertedFile)) {
            throw "Missing converted inference artifact: $convertedFile"
        }
        Push-Location "$OfficialRoot\toolbench\tooleval"
        try {
            & $Python "eval_pass_rate.py" `
                --converted_answer_path $convertedBase `
                --save_path $passMethod `
                --reference_model $label `
                --test_ids "$OfficialRoot\solvable_queries\test_query_ids" `
                --evaluate_times 3 `
                --test_set $group
        }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "StableToolEval SoPR failed for $group" }

        if ($FacModelPath) {
            & $Python "$OfficialRoot\toolbench\tooleval\fac_eval.py" `
                --model_path $FacModelPath `
                --evaluation_path $convertedFile `
                --output_path (Join-Path $facMethod "$group.csv") `
                --ids "$OfficialRoot\solvable_queries\test_query_ids\$group.json"
            if ($LASTEXITCODE -ne 0) { throw "StableToolBench FAC failed for $group" }
        }
    }

    if ($Method -ne "cot") {
        if (
            -not (Test-Path -LiteralPath (Join-Path $convertedBase $reference)) -or
            -not (Test-Path -LiteralPath (Join-Path $passBase $reference))
        ) {
            throw "Run and evaluate the official CoT baseline before calculating SoWR"
        }
        foreach ($group in $subsets) {
            Push-Location "$OfficialRoot\toolbench\tooleval"
            try {
                & $Python "eval_preference.py" `
                    --converted_answer_path $convertedBase `
                    --reference_model $reference `
                    --output_model $label `
                    --test_ids "$OfficialRoot\solvable_queries\test_query_ids" `
                    --save_path $preferenceRoot `
                    --pass_rate_result_path $passBase `
                    --use_pass_rate true `
                    --evaluate_times 3 `
                    --test_set $group
            }
            finally { Pop-Location }
            if ($LASTEXITCODE -ne 0) { throw "StableToolEval SoWR failed for $group" }
        }
    }
}
finally {
    $env:PYTHONPATH = $priorPythonPath
    $env:API_POOL_FILE = $priorApiPool
    $env:EVAL_MODEL = $priorEvalModel
    $env:EVAL_THINKING_MODE = $priorEvalThinkingMode
    $env:EVAL_TEMPERATURE = $priorEvalTemperature
}

if ($Subset -eq "all") {
    $priorImportPath = $env:PYTHONPATH
    $env:PYTHONPATH = "$projectRoot\src"
    try {
        $importArgs = @(
            "$PSScriptRoot\import_official_results.py",
            "--method", $label,
            "--subsets"
        )
        $importArgs += $subsets
        $importArgs += @(
            "--pass-rate-root", $passBase,
            "--sopr-sowr-judge-model", $EvaluatorModel,
            "--sopr-sowr-thinking-mode", $EvaluatorThinkingMode,
            "--sopr-sowr-temperature", $EvaluatorTemperature,
            "--output", "$OutputRoot\$label`_summary.json"
        )
        if ($Method -ne "cot") {
            $importArgs += @(
                "--preference-root", $preferenceRoot,
                "--reference-model", $reference
            )
        }
        if ($FacModelPath) { $importArgs += @("--fac-root", $facBase) }
        & $Python @importArgs
        if ($LASTEXITCODE -ne 0) { throw "Official M3 result import failed" }
    }
    finally { $env:PYTHONPATH = $priorImportPath }
}

Write-Output "Official StableToolBench evaluation complete: $label"
