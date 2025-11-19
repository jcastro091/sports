<#  refresh_weekly.ps1
    Weekly refresh: train model -> build strong config -> (optional) restart runner
    Creates .\weekly_artifacts\YYYYMMDD_HHMMSS\ and writes all logs/artifacts there.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$EngineRoot,                         # e.g. "C:\Users\Jcast\Documents\alpha_signal_engine"

  [Parameter(Mandatory=$true)]
  [string]$Dataset,                            # e.g. "ConfirmedBets - AllObservations.csv" (relative to EngineRoot\data or absolute)

  [string]$EnvFile = ".env",                   # env file in EngineRoot or absolute

  [string]$PythonExe = "python",

  [switch]$RestartRunner,                      # if set, restart sports runner
  [switch]$ForceBalancedBins                   # pass through to strong config builder
)

function Write-Info($msg){ Write-Host "[INFO ] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)  { Write-Host "[ OK  ] $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[WARN ] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[FAIL ] $msg" -ForegroundColor Red }

function Resolve-PathSafe([string]$p){
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path -Path $EngineRoot -ChildPath $p)
}

$ErrorActionPreference = 'Stop'

if (!(Test-Path $EngineRoot)) { Write-Err "EngineRoot not found: $EngineRoot"; exit 1 }

$ts        = Get-Date -Format 'yyyyMMdd_HHmmss'
$rootDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$wkDir     = Join-Path $rootDir "weekly_artifacts"
$outDir    = Join-Path $wkDir  $ts

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Write-Info "Artifacts: $outDir"

Push-Location $EngineRoot
try {
  # --- Load env file (if present) -------------------------------------------
  $envPath = Resolve-PathSafe $EnvFile
  if (Test-Path $envPath) {
    Write-Info "Loading env: $envPath"
    Get-Content $envPath | ForEach-Object {
      if ($_ -match '^\s*#') { return }
      if ($_ -match '^\s*$') { return }
      $parts = $_.Split('=',2)
      if ($parts.Count -eq 2) {
        $k = $parts[0].Trim()
        $v = $parts[1].Trim()
        if ($k) { [System.Environment]::SetEnvironmentVariable($k, $v) }
      }
    }
  } else {
    Write-Warn "Env file not found ($envPath); continuing without it."
  }

  # --- Step 1: Train model ---------------------------------------------------
  $trainLog = Join-Path $outDir 'train_model.log'

  # 5.1-safe dataset resolution (no ternary)
  $datasetUnderData = Resolve-PathSafe (Join-Path 'data' $Dataset)
  if (Test-Path $datasetUnderData) {
    $datasetArg = $datasetUnderData
  } else {
    $datasetArg = Resolve-PathSafe $Dataset
  }

  Write-Info "Training with dataset: $datasetArg"
  & $PythonExe -m src.train_model --dataset "$datasetArg" 2>&1 | Tee-Object -FilePath $trainLog
  Write-Ok "Training complete. Log -> $trainLog"

  # Copy recent model artifacts if present
  $modelDirGuess = Join-Path $EngineRoot 'results'
  if (Test-Path $modelDirGuess) {
    $latestModel = Get-ChildItem $modelDirGuess -Recurse -File -Include *.pkl,*.joblib,*.onnx,*.json |
                   Sort-Object LastWriteTime -Descending | Select-Object -First 10
    if ($latestModel) {
      Write-Info "Copying recent model artifacts to artifacts folder…"
      $latestModel | Copy-Item -Destination $outDir -Force
    } else {
      Write-Warn "No model artifacts found under $modelDirGuess (this may be fine if training only logs)."
    }
  }

  # --- Step 2: Build strong config ------------------------------------------
  $strongOut = Join-Path $outDir 'strong_config.json'
  $strongLog = Join-Path $outDir 'build_strong.log'

  $strongArgs = @("build_strong_config.py","--output","$strongOut")
  if ($ForceBalancedBins.IsPresent) { $strongArgs += "--force-balanced-bins" }

  Write-Info "Building strong config -> $strongOut"
  & $PythonExe @strongArgs 2>&1 | Tee-Object -FilePath $strongLog
  if (Test-Path $strongOut) {
    Write-Ok "Strong config written -> $strongOut"
  } else {
    Write-Warn "Strong config not found after run (expected at $strongOut). Check $strongLog"
  }

  # --- Step 3: Optionally restart runner ------------------------------------
  if ($RestartRunner.IsPresent) {
    Write-Info "Restarting sports runner…"

    # Best-effort stop existing python running sports*.py
    $procs = Get-Process python -ErrorAction SilentlyContinue
    if ($procs) {
      foreach ($p in $procs) {
        try {
          $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
          if ($cmd -and ($cmd -match 'sports.*\.py')) {
            Write-Info "Stopping PID $($p.Id)"
            Stop-Process -Id $p.Id -Force
          }
        } catch { }
      }
    }

    $runner = Join-Path $rootDir 'sports21_fixed.py'   # adjust if needed
    if (!(Test-Path $runner)) {
      Write-Warn "Runner not found at $runner; searching for sports21*.py next to this script…"
      $candidate = Get-ChildItem $rootDir -Filter 'sports21*.py' | Select-Object -First 1
      if ($candidate) { $runner = $candidate.FullName }
    }

    if (Test-Path $runner) {
      $runLog = Join-Path $outDir 'sports21.log'
      $errLog = Join-Path $outDir 'sports21.err'
      Write-Info "Starting: $runner"
      Start-Process -FilePath $PythonExe `
                    -ArgumentList "`"$runner`"" `
                    -WorkingDirectory $rootDir `
                    -RedirectStandardOutput $runLog `
                    -RedirectStandardError  $errLog `
                    -WindowStyle Hidden | Out-Null
      Write-Ok "Runner started. Logs -> $runLog / $errLog"
    } else {
      Write-Err "Couldn’t locate sports runner file. Skipping restart."
    }
  } else {
    Write-Info "RestartRunner not set; skipping runner restart."
  }

} catch {
  Write-Err $_.Exception.Message
  throw
} finally {
  Pop-Location
}

Write-Ok "Weekly refresh done. Artifacts at: $outDir"
