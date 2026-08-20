<#
Prodinno ML Workshop — Windows setup script.
Run from the repo root in a plain Windows Terminal / PowerShell window:

    .\setup.ps1

This creates a local virtual environment (.venv), installs every dependency
declared in pyproject.toml, and registers a Jupyter kernel named
"prodinno-ml-workshop" so all notebooks in every folder can select it.
#>

$ErrorActionPreference = "Stop"

Write-Host "== Prodinno ML Workshop setup ==" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host ".venv already exists, reusing it." -ForegroundColor Yellow
}

$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

Write-Host "Installing project dependencies from pyproject.toml..." -ForegroundColor Yellow
& $venvPython -m pip install -e .

Write-Host "Registering Jupyter kernel 'prodinno-ml-workshop'..." -ForegroundColor Yellow
& $venvPython -m ipykernel install --user --name prodinno-ml-workshop --display-name "Prodinno ML Workshop"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the environment in new terminals with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then open any notebook and pick the 'Prodinno ML Workshop' kernel."
