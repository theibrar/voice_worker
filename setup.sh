#!/usr/bin/env bash
# ==============================================================================
# Enterprise Voice AI GPU Node - Automated One-Click Installer
# Hardware Target: 1x NVIDIA RTX 5060 Ti (16GB VRAM) | Intel Xeon E5-2673 v4
# Public IP: 184.144.154.180
# Stack: Parakeet TDT 0.6B INT8 -> Qwen 4B/7B -> Kokoro-82M (+ Silero VAD v5)
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
echo "    Target GPU : 1x NVIDIA RTX 5060 Ti (16GB VRAM)                            "
echo "    CPU        : Intel Xeon E5-2673 v4 (40 vCPUs, 96.5GB RAM)                 "
echo "    Public IP  : 184.144.154.180                                              "
echo "    Pipeline   : Parakeet TDT 0.6B -> Qwen3-4B / Qwen2.5 -> Kokoro-82M        "
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
    libespeak-ng-dev \
    libcublas-12-0 || true

# 3. Configure Environment & API Key
echo -e "${GREEN}[3/6] Setting Up Secure Environment...${NC}"
DEFAULT_KEY="sk-ibrasoft-gpu-voice"
PUBLIC_IP="184.144.154.180"

cat <<EOF > .env
GPU_API_KEY=${DEFAULT_KEY}
PUBLIC_IP=${PUBLIC_IP}
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
GPU_MEM_UTIL=0.50
MAX_MODEL_LEN=2048
STT_MODEL_SIZE=distil-large-v3
PORT_VLLM=56137
PORT_TTS=56209
PORT_STT=56546
PORT_VAD=56756
PORT_UI=56081
CUDA_MODULE_LOADING=LAZY
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_USE_V1=0
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
KOKORO_MODEL_PATH=/root/voice_worker/models/kokoro-v0_19.onnx
KOKORO_VOICES_PATH=/root/voice_worker/models/voices.bin
EOF

echo -e "${GREEN}✓ Environment configured with secure API key.${NC}"

# 4. Install Python AI Libraries & llama.cpp Server
echo -e "${GREEN}[4/6] Installing PyTorch, vLLM, Faster-Whisper, Kokoro, Silero, Gradio, & llama.cpp...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 || true

# Register NVIDIA python libs directly in /usr/local/lib and system linker (fixes libcublas.so.12)
find /usr/local/lib/ -name "libcublas*.so*" -exec ln -sf {} /usr/local/lib/ \; 2>/dev/null || true
find /usr/local/lib/ -name "libcudnn*.so*" -exec ln -sf {} /usr/local/lib/ \; 2>/dev/null || true
find /usr/local/lib/ -name "libcudart*.so*" -exec ln -sf {} /usr/local/lib/ \; 2>/dev/null || true
echo "/usr/local/lib" > /etc/ld.so.conf.d/00-local.conf
ldconfig 2>/dev/null || true

# 5. Download Neural Model Weights (Kokoro, Parakeet, Qwen)
echo -e "${GREEN}[5/6] Downloading & Verifying Model Weights...${NC}"
mkdir -p models/parakeet models/llm

# A. Kokoro-82M ONNX & Voices (v1.0 Multi-Language: 54 Voices)
if [ ! -f "models/kokoro-v1.0.onnx" ]; then
    echo "Downloading Kokoro-82M v1.0 ONNX Multi-Language Model (320MB)..."
    curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o models/kokoro-v1.0.onnx || true
fi

if [ ! -f "models/voices-v1.0.bin" ] || [ $(wc -c < "models/voices-v1.0.bin" 2>/dev/null || echo 0) -lt 25000000 ]; then
    echo "Downloading Kokoro 54-voice multi-language pack (voices-v1.0.bin)..."
    rm -f models/voices-v1.0.bin
    curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o models/voices-v1.0.bin || true
fi

# Fallback v0_19 weights
if [ ! -f "models/kokoro-v0_19.onnx" ]; then
    curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx -o models/kokoro-v0_19.onnx || true
fi
if [ ! -f "models/voices.bin" ]; then
    curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin -o models/voices.bin || true
fi

# B. Parakeet TDT 0.6B v3 INT8
echo "Downloading Parakeet TDT 0.6B INT8 components..."
PARAKEET_URL="https://huggingface.co/aoiandroid/Parakeet-TDT-0.6B-v3-LiteRT-INT8/resolve/main"

if [ ! -f "models/parakeet/parakeet-encoder.tflite" ]; then
    echo "  • parakeet-encoder.tflite (~567MB)..."
    wget -q --show-progress -c "${PARAKEET_URL}/parakeet-encoder.tflite" -O models/parakeet/parakeet-encoder.tflite || true
fi

if [ ! -f "models/parakeet/parakeet-decoder-joint.tflite" ]; then
    echo "  • parakeet-decoder-joint.tflite (~17MB)..."
    wget -q --show-progress -c "${PARAKEET_URL}/parakeet-decoder-joint.tflite" -O models/parakeet/parakeet-decoder-joint.tflite || true
fi

if [ ! -f "models/parakeet/vocab.json" ]; then
    echo "  • vocab.json..."
    wget -q -c "${PARAKEET_URL}/vocab.json" -O models/parakeet/vocab.json || true
fi

if [ ! -f "models/parakeet/config.json" ]; then
    echo "  • config.json..."
    wget -q -c "${PARAKEET_URL}/config.json" -O models/parakeet/config.json || true
fi

# C. Qwen3-4B Q4_K_M GGUF
if [ ! -f "models/llm/Qwen3-4B-Q4_K_M.gguf" ]; then
    echo "Downloading Qwen3-4B-Q4_K_M.gguf (~2.5GB)..."
    wget -q --show-progress -c "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf" -O models/llm/Qwen3-4B-Q4_K_M.gguf || true
fi

echo -e "${GREEN}✓ All model assets downloaded and cached in ./models/${NC}"

# 6. Generate Production Endpoints File
cat <<EOF > ENDPOINTS.txt
==============================================================================
   APEX ENTERPRISE GPU VOICE AI CLUSTER - PRODUCTION ENDPOINTS
   Hardware: 1x NVIDIA RTX 5060 Ti (16GB VRAM) | Intel Xeon E5-2673 v4
==============================================================================

Public IP: ${PUBLIC_IP}
API Key  : ${DEFAULT_KEY}

1. vLLM / LLM OpenAI Engine (Port 8000)
   Public URL : http://${PUBLIC_IP}:${PORT_VLLM}/v1
   Model      : Qwen3-4B / Qwen2.5-7B-Instruct-AWQ

2. Kokoro-82M Streaming Neural TTS (Port 8088)
   Public URL : http://${PUBLIC_IP}:${PORT_TTS}
   Voices     : Full voice pack (af_bella, am_michael, am_adam, af_sarah, bf_emma)
   Features   : Free-form style tags, SSML <break>, volume gain, <50ms TTFA

3. Fast Streaming STT Engine (Port 8030)
   Public URL : http://${PUBLIC_IP}:${PORT_STT}
   Model      : Faster-Whisper distil-large-v3 (CUDA float16)

4. Silero VAD v5 Controller (Port 8090)
   Public URL : http://${PUBLIC_IP}:${PORT_VAD}
   Spec       : 16kHz, 512 samples / 32ms frame chunking

5. Gradio Live Interactive Playground (Port 7860)
   Public URL : http://${PUBLIC_IP}:${PORT_UI}
==============================================================================
EOF

# 7. Launch All Services via tmux
echo -e "${GREEN}[6/6] Launching All 5 GPU AI Engines in Background...${NC}"
fuser -k 8000/tcp 8030/tcp 8088/tcp 8090/tcp 7860/tcp 2>/dev/null || true
tmux kill-session -t voice-worker 2>/dev/null || true
tmux new-session -d -s voice-worker "export VLLM_USE_FLASHINFER_SAMPLER=0; export VLLM_USE_V1=0; python3 master_orchestrator.py"

sleep 3

cat ENDPOINTS.txt

echo -e "${CYAN}"
echo "=============================================================================="
echo " 🎉 ALL GPU SERVICES ARE RUNNING IN THE BACKGROUND!"
echo "    Inspect live logs anytime with: tmux attach -t voice-worker"
echo "    Or test your mic in your browser at:"
echo "    👉 http://${PUBLIC_IP}:45227"
echo "=============================================================================="
echo -e "${NC}"
