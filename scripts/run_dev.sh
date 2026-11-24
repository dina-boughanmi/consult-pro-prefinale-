#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_dev.sh — lance tout le projet ConsultPro en local en une commande.
#
# Usage :
#   chmod +x run_dev.sh   (une seule fois)
#   ./run_dev.sh
#
# A placer à la racine du repo (au même niveau que backend/ et frontend/).
# Ctrl+C arrête proprement le backend ET le frontend.
# ─────────────────────────────────────────────────────────────────────
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [ -z "${CONSULTPRO_DATABASE_URL:-}" ]; then
    echo "ERREUR: Configure CONSULTPRO_DATABASE_URL vers une base PostgreSQL avant de lancer l'application."
    echo 'Exemple: export CONSULTPRO_DATABASE_URL="postgresql://consultpro:mot_de_passe@localhost:5432/consultpro"'
    exit 1
fi

echo "═══════════════════════════════════════════"
echo "  ConsultPro — démarrage environnement local"
echo "═══════════════════════════════════════════"

# ── 1. Backend : venv + dépendances ───────────────────────────────────
cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
    echo "▶ Création du venv Python..."
    python3 -m venv venv
fi

source venv/bin/activate

# On installe seulement si les deps ne sont pas déjà là (fichier marqueur)
if [ ! -f "venv/.requirements_installed" ]; then
    echo "▶ Installation des dépendances backend..."
    python -m pip install --upgrade pip -q
    python -m pip install -r requirements.txt
    touch venv/.requirements_installed
    echo "✅ Dépendances backend installées."
else
    echo "✅ Dépendances backend déjà installées (skip)."
fi

mkdir -p uploads

# ── 2. Variables d'environnement (test local, modèle Whisper public) ──
export CONSULTPRO_WHISPER_MODEL_PATH="${CONSULTPRO_WHISPER_MODEL_PATH:-base}"
export CONSULTPRO_ASR_DEVICE="${CONSULTPRO_ASR_DEVICE:-cpu}"
export CONSULTPRO_ASR_COMPUTE_TYPE="${CONSULTPRO_ASR_COMPUTE_TYPE:-int8}"
export CONSULTPRO_ASR_CPU_THREADS="${CONSULTPRO_ASR_CPU_THREADS:-2}"
export CONSULTPRO_ASR_BEAM_SIZE="${CONSULTPRO_ASR_BEAM_SIZE:-1}"
export CONSULTPRO_UPLOAD_DIR="${CONSULTPRO_UPLOAD_DIR:-$BACKEND_DIR/uploads}"

echo ""
echo "ℹ️  Modèle Whisper utilisé : $CONSULTPRO_WHISPER_MODEL_PATH"
echo "   (pour utiliser ton modèle fine-tuné converti, exporte"
echo "    CONSULTPRO_WHISPER_MODEL_PATH avant de lancer ce script)"
echo ""

# ── 3. Lancer le backend en arrière-plan ──────────────────────────────
echo "▶ Démarrage du backend FastAPI (http://localhost:8000)..."
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "▶ Démarrage du worker d'analyse (chargement du modèle)..."
python -m app.worker &
WORKER_PID=$!

sleep 3

# ── 4. Frontend : install + run ───────────────────────────────────────
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo "▶ Installation des dépendances frontend (npm install)..."
    npm install
else
    echo "✅ Dépendances frontend déjà installées (skip)."
fi

echo "▶ Démarrage du frontend Vite (http://localhost:5173)..."
npm run dev &
FRONTEND_PID=$!

# ── 6. Nettoyage propre à l'arrêt (Ctrl+C) ────────────────────────────
cleanup() {
    echo ""
    echo "⏹  Arrêt en cours..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    kill "$WORKER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "✅ Backend et frontend arrêtés."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo "═══════════════════════════════════════════"
echo "  API      : http://localhost:8000"
echo "  Worker   : analyse audio"
echo "  Frontend : http://localhost:5173"
echo "  (Ctrl+C pour tout arrêter)"
echo "═══════════════════════════════════════════"

wait
