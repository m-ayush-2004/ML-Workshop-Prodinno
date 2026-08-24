<#
Prodinno ML Workshop — Windows pre-workshop setup script.
Run from the repo root in a plain Windows Terminal / PowerShell window:

    .\setup.ps1

This checks for WSL 2 and Docker Desktop (needed later for the capstone), installs `uv`
if it isn't already on PATH, then uses `uv sync` to create/update the project's virtual
environment (.venv) from pyproject.toml and registers a Jupyter kernel named
"prodinno-ml-workshop" so all notebooks in every folder can select it.

This script does NOT build or run the capstone's Docker containers — that's a separate,
optional step (see 06_capstone/README.md): once Docker Desktop is installed and running,

    cd 06_capstone
    docker compose up --build
#>

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    # Runs an external command and stops the script with a clear message if it exits non-zero.
    # (Native-command failures don't trigger $ErrorActionPreference="Stop" on their own.)
    param([Parameter(Mandatory)][ScriptBlock]$Command, [Parameter(Mandatory)][string]$FailureMessage)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n$FailureMessage" -ForegroundColor Red
        exit 1
    }
}

Write-Host "== Prodinno ML Workshop setup ==" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. WSL 2 (needed later for Docker Desktop's WSL 2 backend)
# ---------------------------------------------------------------------------
Write-Host "`nChecking WSL 2..." -ForegroundColor Cyan
$wslOk = $false
try {
    wsl --status | Out-Null
    if ($LASTEXITCODE -eq 0) { $wslOk = $true }
} catch {
    $wslOk = $false
}

if ($wslOk) {
    Write-Host "WSL is installed." -ForegroundColor Green
} else {
    Write-Host "WSL 2 was not detected." -ForegroundColor Yellow
    Write-Host "This script will NOT install it automatically (it needs an elevated" -ForegroundColor Yellow
    Write-Host "PowerShell session and may require a restart). To install it yourself:" -ForegroundColor Yellow
    Write-Host "  1. Open PowerShell as Administrator and run:  wsl --install" -ForegroundColor Yellow
    Write-Host "  2. Restart your system if prompted." -ForegroundColor Yellow
    Write-Host "  If that command fails, download the WSL package from:" -ForegroundColor Yellow
    Write-Host "    https://github.com/microsoft/WSL/releases" -ForegroundColor Yellow
    Write-Host "  run the .msi installer, restart, then verify with:  wsl --status" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 2. Docker Desktop (needed later for the capstone)
# ---------------------------------------------------------------------------
Write-Host "`nChecking Docker Desktop..." -ForegroundColor Cyan
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue

if ($dockerCmd) {
    docker --version
    docker compose version
    Write-Host "Running 'docker run hello-world' to confirm the engine is up..." -ForegroundColor Yellow
    try {
        docker run hello-world | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker is installed and running." -ForegroundColor Green
        } else {
            Write-Host "Docker is installed but the daemon doesn't seem to be running -- open Docker Desktop first." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Docker is installed but the daemon doesn't seem to be running -- open Docker Desktop first." -ForegroundColor Yellow
    }
} else {
    Write-Host "Docker Desktop was not detected." -ForegroundColor Yellow
    Write-Host "This script will NOT install it automatically -- it's a large GUI installer" -ForegroundColor Yellow
    Write-Host "that needs to be downloaded and run by hand. To install it yourself:" -ForegroundColor Yellow
    Write-Host "  1. Download 'Docker Desktop for Windows - AMD64' from:" -ForegroundColor Yellow
    Write-Host "       https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "     (AMD64 is correct even on Intel CPUs.)" -ForegroundColor Yellow
    Write-Host "  2. During installation, select the WSL 2 based engine." -ForegroundColor Yellow
    Write-Host "  3. Open Docker Desktop and make sure it's running, then verify with:" -ForegroundColor Yellow
    Write-Host "       docker --version" -ForegroundColor Yellow
    Write-Host "       docker compose version" -ForegroundColor Yellow
    Write-Host "       docker run hello-world" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 3. uv
# ---------------------------------------------------------------------------
Write-Host "`nChecking uv..." -ForegroundColor Cyan
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uvCmd) {
    Write-Host "uv not found -- installing..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # uv installs to a user-local bin directory that may not be on PATH yet in this
    # process -- refresh PATH from the registry so the rest of this script can find it.
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Host "uv was installed but isn't on PATH in this session yet." -ForegroundColor Yellow
        Write-Host "Close and reopen PowerShell, then re-run .\setup.ps1" -ForegroundColor Yellow
        exit 1
    }
}

uv --version
Write-Host "uv is installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Project environment: uv sync (creates/updates .venv from pyproject.toml)
# ---------------------------------------------------------------------------
Write-Host "`nSyncing the project environment with uv (this creates/updates .venv)..." -ForegroundColor Cyan
Invoke-Checked -Command { uv sync } -FailureMessage @"
'uv sync' failed. The most common cause on Windows is a stale, locked .venv --
e.g. an existing .venv created by 'python -m venv' + pip, or a Jupyter
kernel/VS Code Python process still running from it.
Close any running Jupyter/Python processes using this project's .venv, or
delete the .venv folder entirely and re-run .\setup.ps1.
"@

Write-Host "`nRegistering Jupyter kernel 'prodinno-ml-workshop'..." -ForegroundColor Cyan
Invoke-Checked -Command { uv run python -m ipykernel install --user --name prodinno-ml-workshop --display-name "Prodinno ML Workshop" } `
    -FailureMessage "Failed to register the Jupyter kernel -- see the error above."

# ---------------------------------------------------------------------------
# 5. Verification
# ---------------------------------------------------------------------------
Write-Host "`nVerifying the Python environment..." -ForegroundColor Cyan
Invoke-Checked -Command { uv run python -c "import pandas, numpy, sklearn, xgboost, matplotlib, seaborn, plotly, shap, lime; print('Python environment OK')" } `
    -FailureMessage "One or more packages failed to import -- see the error above."

Write-Host "`nVerifying Jupyter..." -ForegroundColor Cyan
Invoke-Checked -Command { uv run jupyter --version } -FailureMessage "Jupyter verification failed -- see the error above."

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run notebook commands with 'uv run', e.g.:" -ForegroundColor Green
Write-Host "    uv run jupyter lab"
Write-Host "Then open any notebook and pick the 'Prodinno ML Workshop' kernel."
Write-Host ""
Write-Host "Docker/WSL not installed yet? See the warnings above -- you only need them" -ForegroundColor Yellow
Write-Host "for the capstone (06_capstone/), not for the numbered topic notebooks." -ForegroundColor Yellow
