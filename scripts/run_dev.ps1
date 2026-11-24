# run_dev.ps1 — lance tout le projet ConsultPro en local en une commande (Windows PowerShell).
#
# Usage (dans le terminal VS Code, en PowerShell) :
#   .\scripts\run_dev.ps1
#
# A placer dans scripts/ à la racine du repo (au même niveau que backend/ et frontend/).
#
# Fonctionnement : ce script ouvre trois fenetres separees pour l'API, le worker
# de transcription et le frontend. Pour tout arreter, ferme les trois fenetres.

$ErrorActionPreference = "Stop"

$RootDir     = Split-Path -Parent $PSScriptRoot
$BackendDir  = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"

if (-not $env:CONSULTPRO_DATABASE_URL) {
    Write-Host "ERREUR: Configure CONSULTPRO_DATABASE_URL vers une base PostgreSQL avant de lancer l'application."
    Write-Host 'Exemple: $env:CONSULTPRO_DATABASE_URL = "postgresql://consultpro:mot_de_passe@localhost:5432/consultpro"'
    exit 1
}

Write-Host "==============================================="
Write-Host "  ConsultPro - demarrage environnement local"
Write-Host "==============================================="

# ── 1. Backend : venv + dependances (verifie/installe depuis cette fenetre) ──
Set-Location $BackendDir

if (-not (Test-Path "venv")) {
    Write-Host "-> Creation du venv Python (3.11 64-bit)..."
    py -3.11 -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERREUR: Python 3.11 introuvable via 'py -3.11'."
        Write-Host "Verifie l'installation avec: py -0"
        Write-Host "Ou remplace '-3.11' par '-3.10' dans ce script si tu utilises 3.10."
        exit 1
    }
}

$DepsMarker = Join-Path $BackendDir "venv\.requirements_installed"
if (-not (Test-Path $DepsMarker)) {
    Write-Host "-> Installation des dependances backend..."
    & "venv\Scripts\python.exe" -m pip install --upgrade pip -q
    & "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    New-Item -ItemType File -Path $DepsMarker | Out-Null
    Write-Host "OK - Dependances backend installees."
} else {
    Write-Host "OK - Dependances backend deja installees (skip)."
}

New-Item -ItemType Directory -Force -Path (Join-Path $BackendDir "uploads") | Out-Null

# ── 2. Variables d'environnement (test local, modele fine-tune) ──
if (-not $env:CONSULTPRO_WHISPER_MODEL_PATH) { $env:CONSULTPRO_WHISPER_MODEL_PATH = Join-Path $RootDir "model\whisper-dental-fr-final-ct2" }
if (-not $env:CONSULTPRO_ASR_DEVICE)         { $env:CONSULTPRO_ASR_DEVICE = "cuda" }
if (-not $env:CONSULTPRO_ASR_COMPUTE_TYPE)    { $env:CONSULTPRO_ASR_COMPUTE_TYPE = "float16" }
if (-not $env:CONSULTPRO_ASR_CPU_THREADS)     { $env:CONSULTPRO_ASR_CPU_THREADS = "0" }
if (-not $env:CONSULTPRO_UPLOAD_DIR)          { $env:CONSULTPRO_UPLOAD_DIR = Join-Path $BackendDir "uploads" }

Write-Host ""
Write-Host "Modele Whisper utilise : $env:CONSULTPRO_WHISPER_MODEL_PATH"
Write-Host ""

# ── 3. Lancer l'API dans une nouvelle fenêtre ─────────────────────────
# On utilise python.exe du venv directement (pas Activate.ps1) : plus robuste,
# pas besoin de policy d'execution particuliere dans la fenetre enfant.
Write-Host "-> Ouverture d'une fenetre pour l'API FastAPI (http://localhost:8000)..."

$BackendCmd = @"
Set-Location '$BackendDir'
`$env:CONSULTPRO_UPLOAD_DIR = '$env:CONSULTPRO_UPLOAD_DIR'
Write-Host '=== ConsultPro Backend ===' -ForegroundColor Cyan
& '$BackendDir\venv\Scripts\python.exe' -m uvicorn app.main:app --reload --port 8000
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd

# ── 4. Lancer le worker dans une nouvelle fenêtre ────────────────────
$WorkerCmd = @"
Set-Location '$BackendDir'
Write-Host '=== ConsultPro Analysis Worker ===' -ForegroundColor Yellow
& '$BackendDir\venv\Scripts\python.exe' -m app.worker
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $WorkerCmd

# ── 5. Frontend : install si besoin (verifie depuis cette fenetre) ────
Set-Location $FrontendDir

if (-not (Test-Path "node_modules")) {
    Write-Host "-> Installation des dependances frontend (npm install)..."
    npm install
} else {
    Write-Host "OK - Dependances frontend deja installees (skip)."
}

# ── 6. Lancer le frontend dans une nouvelle fenêtre ──────────────────
Write-Host "-> Ouverture d'une fenetre pour le frontend Vite (http://localhost:5173)..."

$FrontendCmd = @"
Set-Location '$FrontendDir'
Write-Host '=== ConsultPro Frontend ===' -ForegroundColor Green
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host ""
Write-Host "==============================================="
Write-Host "  API      : http://localhost:8000  (fenetre separee)"
Write-Host "  Worker   : analyse audio (fenetre separee)"
Write-Host "  Frontend : http://localhost:5173  (fenetre separee)"
Write-Host "  Pour tout arreter : ferme les trois nouvelles fenetres"
Write-Host "  (ou Ctrl+C dans chacune)"
Write-Host "==============================================="
Write-Host ""
Write-Host "Cette fenetre-ci peut etre fermee, les services"
Write-Host "frontend continueront de tourner dans leurs fenetres respectives."