#!/usr/bin/env bash
# ==============================================================================
# Enterprise Voice AI GPU Node - Automated One-Click Installer
# Hardware Target: NVIDIA RTX 3090 (24GB VRAM)
# Public IP: 212.93.107.107
# ==============================================================================

set -e

# Terminal Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}"
echo "=============================================================================="
echo "    🎙️  ENTERPRISE GPU VOICE AI WORKER - AUTOMATED INSTALLER                  "
echo "    Target GPU : NVIDIA RTX 3090 (24GB VRAM)                                  "
echo "    Public IP  : 212.93.107.107                                               "
echo "=============================================================================="
echo -e "${NC}"

# 1. Check NVIDIA GPU
echo -e "${GREEN}[1/6] Detecting NVIDIA GPU Hardware...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU verified.${NC}"
else
    echo -e "${RED}❌ NVIDIA GPU not found. Please run on an NVIDIA GPU instance.${NC}"
    exit 1
fi

# 2. Install System Dependencies
echo -e "${GREEN}[2/6] Installing Audio & DSP System Libraries...${NC}"
apt-get update -y
apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    git \
    wget \
    curl \
    tmux \
    python3-pip \
    python3-dev \
    build-essential

# 3. Configure API Key
echo -e "${GREEN}[3/6] Setting Up Secure API Key...${NC}"
DEFAULT_KEY="sk-ibrasoft-gpu-voice"

if [ -z "$GPU_API_KEY" ]; then
    read -p "Enter your custom GPU API Key (press Enter to use '$DEFAULT_KEY'): " USER_KEY
    GPU_API_KEY="${USER_KEY:-$DEFAULT_KEY}"
fi

cat <<EOF > .env
GPU_API_KEY=${GPU_API_KEY}
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
STT_MODEL_SIZE=distil-large-v3
GPU_MEM_UTIL=0.65
KOKORO_MODEL_PATH=/root/voice_worker/models/kokoro-v0_19.onnx
KOKORO_VOICES_PATH=/root/voice_worker/models/voices.bin
EOF

echo -e "${GREEN}✓ API Key set: ${CYAN}${GPU_API_KEY}${NC}"

# 4. Install Python AI Libraries
echo -e "${GREEN}[4/6] Installing PyTorch, vLLM, Faster-Whisper, Kokoro, Silero, & Gradio...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt

# 5. Download Kokoro Neural Model & Voices
echo -e "${GREEN}[5/6] Downloading Kokoro-82M ONNX Neural Audio Weights & Voices...${NC}"
mkdir -p models

if [ ! -f "models/kokoro-v0_19.onnx" ]; then
    echo "Downloading kokoro-v0_19.onnx (320MB)..."
    wget -q --show-progress -c https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx -O models/kokoro-v0_19.onnx
fi

if [ ! -f "models/voices.bin" ]; then
    echo "Downloading voices.bin (28MB)..."
    wget -q --show-progress -c https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin -O models/voices.bin
fi

echo -e "${GREEN}✓ Neural audio models cached in ./models/${NC}"

# 6. Generate Endpoints File for Super Admin
cat <<EOF > ENDPOINTS.txt
==============================================================================
   APEX ENTERPRISE GPU VOICE AI CLUSTER - PRODUCTION ENDPOINTS
==============================================================================

Public IP: 212.93.107.107
API Key  : ${GPU_API_KEY}

1. vLLM OpenAI-Compatible LLM Engine (Port 8000)
   Base URL : http://212.93.107.107:41091/v1
   Model    : Qwen/Qwen2.5-7B-Instruct

2. Kokoro-82M Streaming Neural TTS Engine (Port 8088)
   Base URL : http://212.93.107.107:41438
   Voices   : af_bella, af_sarah, am_adam, am_michael, bf_emma

3. Fast Streaming STT Transcriber with Denoising (Port 8030)
   Base URL : http://212.93.107.107:41182
   Model    : distil-large-v3 / Parakeet

4. Silero VAD & Barge-In Controller (Port 8090)
   Base URL : http://212.93.107.107:41423

5. Gradio Real-Time Audio Playground & Prosody Tuner (Port 7860)
   Web UI   : http://212.93.107.107:41064
==============================================================================
EOF

# 7. Launch All Services via tmux
echo -e "${GREEN}[6/6] Launching All 5 GPU AI Engines in Background...${NC}"
tmux kill-session -t voice-worker 2>/dev/null || true
tmux new-session -d -s voice-worker "python3 master_orchestrator.py"

sleep 3

cat ENDPOINTS.txt

echo -e "${CYAN}"
echo "=============================================================================="
echo " 🎉 ALL GPU SERVICES ARE NOW RUNNING IN THE BACKGROUND!"
echo "    You can inspect live logs anytime with: tmux attach -t voice-worker"
echo "    Or test your mic right now in your browser at:"
echo "    👉 http://212.93.107.107:41064"
echo "=============================================================================="
echo -e "${NC}"
