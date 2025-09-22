#!/bin/bash

# MemVec Development Setup + Server Launcher

set -euo pipefail

echo "🚀 Setting up MemVec development environment..."

# Ensure Python 3 is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 is required but not found. Please install Python 3."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    echo "🔧 Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

PY_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

# Upgrade pip (safe and quick)
echo "⬆️  Ensuring pip is up to date..."
"$PIP_BIN" install --upgrade pip >/dev/null

# Check key deps; install requirements if any are missing
missing=0
for pkg in fastapi uvicorn; do
    if ! "$PY_BIN" -c "import $pkg" >/dev/null 2>&1; then
        missing=1
        break
    fi
done

if [ "$missing" -eq 1 ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    "$PIP_BIN" install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "✅ Required packages already installed."
fi

# Optionally start Redis via Docker Compose if available
if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
    echo "🐳 Starting Redis container (docker-compose up -d)..."
    docker-compose up -d
    echo "✅ Redis is running on localhost:6379"
else
    echo "⚠️  Docker or docker-compose not found. Skipping Redis startup."
fi

# Create .env from template if missing
if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
    echo "📝 Creating .env from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "✏️  Edit .env with your configuration values."
fi

echo "🚀 Launching FastAPI server (uvicorn api.main:app) on http://0.0.0.0:8000 ..."
exec "$PY_BIN" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
