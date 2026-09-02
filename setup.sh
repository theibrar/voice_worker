#!/usr/bin/env bash
# ==============================================================================
# Enterprise Voice AI GPU Node - Automated One-Click Installer
# Hardware Target: NVIDIA RTX 3060 (12GB VRAM) | AMD EPYC 7502P
# Public IP: 173.185.79.174
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
echo "    Target GPU : NVIDIA RTX 3060 (12GB VRAM) | AMD EPYC 7502P                 "
echo "    Public IP  : 173.185.79.174                                               "
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
killall unattended-upgr apt apt-get 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

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
    build-essential \
    espeak-ng \
    libespeak-ng-dev \
    libespeak-ng1

# 3. Configure Environment Variables
echo -e "${GREEN}[3/6] Setting Up Configuration...${NC}"
DEFAULT_KEY="sk-ibrasoft-gpu-voice"
GPU_API_KEY="${GPU_API_KEY:-$DEFAULT_KEY}"

cat <<EOF > .env
GPU_API_KEY=${GPU_API_KEY}
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
STT_MODEL_SIZE=distil-large-v3
GPU_MEM_UTIL=0.65
KOKORO_MODEL_PATH=/root/voice_worker/models/kokoro-v0_19.onnx
KOKORO_VOICES_PATH=/root/voice_worker/models/voices.bin
EOF

echo -e "${GREEN}✓ Environment configured.${NC}"

# 4. Install Python AI Libraries
echo -e "${GREEN}[4/6] Installing Python AI Stack...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install --no-cache-dir onnxruntime-gpu==1.19.0

# Setup NVIDIA library links for ONNX and CTranslate2
echo -e "${GREEN}Configuring NVIDIA CUDA dynamic libraries...${NC}"
for dir in /usr/local/cuda/lib64 $(find /usr/local/lib/python3.10/dist-packages/nvidia/ -name "lib" -type d 2>/dev/null); do
    for f in $dir/*.so*; do
        if [ -f "$f" ]; then
            base=$(basename "$f")
            ln -sf "$f" "/usr/lib/$base" 2>/dev/null || true
            if [[ "$base" == *.so.12* ]]; then
                alias11=${base/.so.12/.so.11}
                alias10=${base/.so.12/.so.10}
                ln -sf "$f" "/usr/lib/$alias11" 2>/dev/null || true
                ln -sf "$f" "/usr/lib/$alias10" 2>/dev/null || true
            fi
        fi
    done
done
ldconfig 2>/dev/null || true

# 5. Download Kokoro Neural Model & Voices
echo -e "${GREEN}[5/6] Checking Kokoro-82M ONNX Neural Audio Weights & Voices...${NC}"
mkdir -p models

if [ ! -f "models/kokoro-v0_19.onnx" ]; then
    echo "Downloading kokoro-v0_19.onnx (320MB)..."
    wget -q --show-progress -c https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx -O models/kokoro-v0_19.onnx
fi

if [ ! -f "models/voices.bin" ]; then
    echo "Downloading voices.bin (28MB)..."
    wget -q --show-progress -c https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin -O models/voices.bin
fi

echo -e "${GREEN}✓ Neural audio models ready in ./models/${NC}"

# 6. Generate Endpoints File
cat <<EOF > ENDPOINTS.txt
==============================================================================
   APEX ENTERPRISE GPU VOICE AI CLUSTER - PRODUCTION ENDPOINTS
==============================================================================

Public IP: 173.185.79.174
API Key  : ${GPU_API_KEY}

1. vLLM OpenAI-Compatible LLM Engine (Port 8000)
   Base URL : http://173.185.79.174:46409/v1
   Model    : Qwen/Qwen2.5-7B-Instruct-AWQ

2. Kokoro-82M Streaming Neural TTS Engine (Port 8088)
   Base URL : http://173.185.79.174:47830
   Voices   : af_bella, af_sarah, am_adam, am_michael, bf_emma

3. Fast Streaming STT Transcriber with Denoising (Port 8030)
   Base URL : http://173.185.79.174:46819
   Model    : distil-large-v3

4. Silero VAD & Barge-In Controller (Port 8090)
   Base URL : http://173.185.79.174:49760

5. Gradio Real-Time Audio Playground (Port 7860)
   Web UI   : http://173.185.79.174:47761
==============================================================================
EOF

# 7. Launch All Services via tmux
echo -e "${GREEN}[6/6] Launching All 5 GPU AI Engines in Background...${NC}"
pkill -9 -f "python3" 2>/dev/null || true
tmux kill-session -t voice-worker 2>/dev/null || true
tmux new-session -d -s voice-worker "python3 master_orchestrator.py"

sleep 3

cat ENDPOINTS.txt

echo -e "${CYAN}"
echo "=============================================================================="
echo " 🎉 ALL GPU SERVICES ARE NOW RUNNING IN THE BACKGROUND!"
echo "    You can inspect live logs anytime with: tmux attach -t voice-worker"
echo "    Or test in your browser at:"
echo "    👉 http://173.185.79.174:47761"
echo "=============================================================================="
echo -e "${NC}"
