#!/usr/bin/env bash
# ==============================================================================
# Quick Start / Launcher for GPU Voice Worker Node
# Domain: server.ibrasoft.com
# ==============================================================================

set -e

echo "🚀 Starting Enterprise AI Voice GPU Node (server.ibrasoft.com)..."

# Ensure .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Ensure models directory exists
mkdir -p models ssl

# Launch with GPU acceleration
docker compose -f docker-compose.gpu.yml up -d --build

echo ""
echo "✅ All GPU services started successfully!"
echo "Run 'docker ps' to see active containers."
echo "Run 'docker compose -f docker-compose.gpu.yml logs -f voice-agent-gpu' to view live call logs."
