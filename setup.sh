#!/bin/bash

# MemVec Development Setup Script
# This script sets up the development environment for MemVec

set -e

echo "🚀 Setting up MemVec development environment..."

# Check if Python 3 is available
if ! command -v py &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    py -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if Docker is available for Redis
if command -v docker &> /dev/null; then
    echo "🐳 Starting Redis container..."
    docker-compose up -d
    echo "✅ Redis is running on localhost:6379"
else
    echo "⚠️  Docker not found. Please install Docker to run Redis locally."
    echo "   Alternatively, install Redis directly or use a cloud Redis instance."
fi

# Copy environment template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✏️  Please edit .env with your actual configuration values."
fi

echo ""
echo "✅ MemVec development environment is ready!"
echo ""
echo "🔄 To activate the environment in future sessions:"
echo "   source .venv/bin/activate"
echo ""
echo "🧪 To run tests:"
echo "   make test"
echo ""
echo "🚀 To start development:"
echo "   python -m src.main"
