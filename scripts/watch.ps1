# Re-run one test every time tokenizer.py is saved.
#
#   .\scripts\watch.ps1                                   # default target
#   .\scripts\watch.ps1 tests/test_train.py::test_vocab_size_is_exact
#   .\scripts\watch.ps1 tests/test_roundtrip.py -x        # extra pytest args
#
# Watch settings live in [tool.pytest-watcher] in pyproject.toml. Ctrl+C stops.

param(
    [string]$Test = "tests/test_train.py::test_vocab_size_exactly_256_is_legal",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$root = Split-Path -Parent $PSScriptRoot
$scripts = Join-Path $root ".venv\Scripts"
$ptw = Join-Path $scripts "ptw.exe"
$pytest = Join-Path $scripts "pytest.exe"

if (-not (Test-Path $ptw)) {
    Write-Error "ptw not found in .venv. Run: .venv\Scripts\python -m pip install -e "".[dev]"""
    exit 1
}

# pytest resolves the test id relative to the repo root.
Push-Location $root
try {
    & $ptw --runner $pytest $root $Test @PytestArgs
} finally {
    Pop-Location
}
