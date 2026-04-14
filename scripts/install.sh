#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  AG Vision (HybridFace) — Automated Setup Script
#  Compatible with: macOS (Apple Silicon M1/M2/M3) · macOS Intel
#  Author: Léonard Manzanera — Avril 2026
# ══════════════════════════════════════════════════════════════════

set -e  # Exit immediately on error

echo ""
echo "══════════════════════════════════════════════"
echo "  🚀  AG Vision — Setup Automatisé"
echo "══════════════════════════════════════════════"
echo ""

# ── Detect OS & architecture ──────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "[i] Système détecté : $OS ($ARCH)"

# ── Check Python 3 ───────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "[✗] Python 3 est introuvable. Veuillez l'installer depuis https://www.python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "[✓] Python $PYTHON_VERSION détecté."

# ── Check / Install CMake (required for dlib) ────────────────────
echo ""
echo "[*] Vérification de CMake (requis pour compiler dlib)..."

if ! command -v cmake &> /dev/null; then
    echo "[!] CMake introuvable."
    if [ "$OS" = "Darwin" ] && command -v brew &> /dev/null; then
        echo "[*] Installation de CMake via Homebrew..."
        brew install cmake
    else
        echo ""
        echo "══════════════════════════════════════════════"
        echo "  ⚠️  INSTALLATION MANUELLE REQUISE"
        echo "══════════════════════════════════════════════"
        echo "  CMake n'est pas installé et ne peut pas"
        echo "  être installé automatiquement."
        echo ""
        echo "  → macOS  : brew install cmake"
        echo "  → Ubuntu : sudo apt-get install cmake build-essential"
        echo "  → Windows: https://cmake.org/download/"
        echo ""
        echo "  Relancez ce script après installation."
        echo "══════════════════════════════════════════════"
        exit 1
    fi
fi
echo "[✓] CMake $(cmake --version | head -n1 | awk '{print $3}') disponible."

# ── Virtual environment ───────────────────────────────────────────
echo ""
read -p "[?] Créer un environnement virtuel Python ? (recommandé) [Y/n] " -r VENV_REPLY
VENV_REPLY="${VENV_REPLY:-Y}"

if [[ "$VENV_REPLY" =~ ^[Yy]$ ]]; then
    VENV_DIR="ag_env"
    echo "[*] Création de l'environnement virtuel '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "[✓] Environnement virtuel activé."
    USING_VENV=true
else
    echo "[i] Installation dans l'environnement Python global."
    USING_VENV=false
fi

# ── Upgrade pip + build tools ─────────────────────────────────────
echo ""
echo "[*] Mise à jour de pip et des outils de build..."
python3 -m pip install --upgrade pip setuptools wheel cmake

# ── Install Python dependencies ───────────────────────────────────
echo ""
echo "[*] Installation des dépendances Python..."
echo "    (dlib peut prendre 3-5 minutes à compiler, soyez patient)"
echo ""
python3 -m pip install -r requirements.txt

# ── Download ML models ────────────────────────────────────────────
echo ""
echo "[*] Téléchargement des modèles ML (≈ 450 Mo au total)..."
python3 scripts/setup.py

# ── Final summary ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  ✅  Installation terminée avec succès !"
echo "══════════════════════════════════════════════"
echo ""
echo "  Pour lancer l'application :"
echo ""
if [ "$USING_VENV" = true ]; then
    echo "    source ag_env/bin/activate"
fi
echo "    python3 launcher.py"
echo ""
echo "  Ou directement avec le raccourci :"
if [ "$USING_VENV" = true ]; then
    echo "    source ag_env/bin/activate && ./start"
else
    echo "    ./start"
fi
echo ""
echo "══════════════════════════════════════════════"
