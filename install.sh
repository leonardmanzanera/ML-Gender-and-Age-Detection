#!/bin/bash
# AG Vision - Setup Script for Mac M1

echo "=================================================="
echo "  🚀 AG Vision (Antigravity) setup for Mac M1"
echo "=================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "[!] Python 3 could not be found. Please install Python 3."
    exit 1
fi

# Ask to create a virtual environment
read -p "[?] Create a virtual environment (venv)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "[*] Creating virtual environment (ag_env)..."
    python3 -m venv ag_env
    source ag_env/bin/activate
    echo "[+] Virtual environment activated."
else
    echo "[i] Skipping virtual environment creation."
fi

# Upgrade pip and install build dependencies
echo "[*] Upgrading pip and installing build tools (cmake)..."
python3 -m pip install --upgrade pip
python3 -m pip install cmake

# Install dependencies from requirements.txt
echo "[*] Installing Python dependencies (this may take a few minutes for dlib)..."
python3 -m pip install -r requirements.txt

# Download missing models
echo "[*] Downloading ML models (this may take a few minutes)..."
python3 scripts/setup.py

if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "✅ Setup Complete!"
    echo "To run the application:"
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  source ag_env/bin/activate"
    fi
    echo "  python launcher.py"
    echo "=================================================="
else
    echo "=================================================="
    echo "❌ Setup failed. Please check the error messages above."
    echo "=================================================="
fi
