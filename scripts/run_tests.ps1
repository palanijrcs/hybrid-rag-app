# Phase 19: run every test suite.  From the project root, venv active:
#   .\scripts\run_tests.ps1           # all offline tests + coverage
#   .\scripts\run_tests.ps1 -Quick    # skip model-loading tests
#   .\scripts\run_tests.ps1 -Live     # also real OpenAI/Neo4j smoke test (a few cents)
param([switch]$Quick, [switch]$Live)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$failed = $false

Write-Host "`n=== Backend ===" -ForegroundColor Cyan
Push-Location "$root\backend"
$marker = if ($Quick) { "not slow and not live" } else { "not live" }
python -m pytest -m $marker --cov=app --cov-report=term-missing:skip-covered -q
if ($LASTEXITCODE -ne 0) { $failed = $true }
if ($Live) {
    Write-Host "`n=== Live smoke test ===" -ForegroundColor Cyan
    $env:RUN_LIVE_TESTS = "1"
    python -m pytest -m live -v
    if ($LASTEXITCODE -ne 0) { $failed = $true }
    Remove-Item Env:RUN_LIVE_TESTS
}
Pop-Location

Write-Host "`n=== Frontend ===" -ForegroundColor Cyan
Push-Location "$root\frontend"
python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { $failed = $true }
Pop-Location

if ($failed) { Write-Host "`nSOME TESTS FAILED" -ForegroundColor Red; exit 1 }
Write-Host "`nALL TESTS PASSED" -ForegroundColor Green
