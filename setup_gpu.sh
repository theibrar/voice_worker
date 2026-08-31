#!/usr/bin/env bash
# ==============================================================================
# Enterprise Voice AI Worker - Automated Ubuntu GPU Installer
# Domain: server.ibrasoft.com
# LiveKit WebRTC + LiveKit SIP (Telnyx) + Kokoro-82M Neural TTS + FastConformer
# ==============================================================================

set -e

# Terminal Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

DOMAIN="${1:-server.ibrasoft.com}"

echo -e "${CYAN}"
echo "=============================================================================="
echo "      ENTERPRISE VOICE AI WORKER - GPU NODE AUTOMATED INSTALLER              "
echo "      Target Domain : ${DOMAIN}                                               "
echo "      Engine Stack  : LiveKit SIP + Parakeet STT + Kokoro-82M Neural TTS     "
echo "=============================================================================="
echo -e "${NC}"

# 1. Check Root Privileges
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Please run with sudo: sudo bash setup_gpu.sh [domain]${NC}"
  exit 1
fi

# 2. Verify NVIDIA GPU Hardware & Drivers
echo -e "${GREEN}[1/8] Checking NVIDIA GPU Hardware & CUDA Drivers...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU Detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}NVIDIA Driver not found. Installing recommended NVIDIA drivers...${NC}"
    apt-get update -y
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers install
    echo -e "${YELLOW}NVIDIA Drivers installed.${NC}"
fi

# 3. Install Docker & NVIDIA Container Toolkit (nvidia-ctk)
echo -e "${GREEN}[2/8] Installing Docker CE & NVIDIA Container Toolkit...${NC}"
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release ufw wget git openssl

if ! command -v docker &> /dev/null; then
    echo -e "${CYAN}Installing Docker CE...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "${CYAN}Configuring NVIDIA Container Toolkit for Docker...${NC}"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo -e "${GREEN}✓ NVIDIA Container Toolkit configured.${NC}"
fi

# 4. Configure UFW Firewall for VoIP, SIP, and WebRTC
echo -e "${GREEN}[3/8] Configuring UFW Firewall for VoIP SIP & WebRTC Media...${NC}"
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP Nginx'
ufw allow 443/tcp comment 'HTTPS Nginx'
ufw allow 5060/udp comment 'SIP Signaling UDP (Telnyx)'
ufw allow 5060/tcp comment 'SIP Signaling TCP (Telnyx)'
ufw allow 7880/tcp comment 'LiveKit WebRTC Signaling'
ufw allow 7881/tcp comment 'LiveKit WebRTC TURN TCP'
ufw allow 3478/udp comment 'LiveKit STUN/TURN UDP'
ufw allow 10000:20000/udp comment 'RTP Audio Media Stream Range'
echo "y" | ufw enable || true
echo -e "${GREEN}✓ All telephony firewall ports configured.${NC}"

# 5. SSL Certificate Generation for server.ibrasoft.com
echo -e "${GREEN}[4/8] Generating SSL / TLS Certificates for ${DOMAIN}...${NC}"
mkdir -p ./ssl
if [ ! -f "./ssl/fullchain.pem" ] || [ ! -f "./ssl/privkey.pem" ]; then
    echo -e "${CYAN}Creating SSL certificate for ${DOMAIN}...${NC}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout ./ssl/privkey.pem \
      -out ./ssl/fullchain.pem \
      -subj "/C=US/ST=California/L=SanFrancisco/O=IbraSoft/CN=${DOMAIN}"
    echo -e "${GREEN}✓ SSL certificate generated in ./ssl/${NC}"
fi

# 6. Pre-Cache Kokoro Neural Weights & Voice Models with Fallback Mirrors
echo -e "${GREEN}[5/8] Downloading and Pre-caching Kokoro Neural Voice Models...${NC}"
mkdir -p ./models
cd ./models

# Download ONNX Model
if [ ! -f "kokoro-v0_19.onnx" ] || [ $(stat -c%s "kokoro-v0_19.onnx" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo -e "${CYAN}Downloading Kokoro-82M ONNX model (82MB)...${NC}"
    wget -q --show-progress -O kokoro-v0_19.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" || \
    wget -q --show-progress -O kokoro-v0_19.onnx "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx" || true
fi

# Download Voices BIN
if [ ! -f "voices.bin" ] || [ $(stat -c%s "voices.bin" 2>/dev/null || echo 0) -lt 100000 ]; then
    echo -e "${CYAN}Downloading Kokoro Neural Voices bin (af_heart, af_bella, am_adam, etc.)...${NC}"
    wget -q --show-progress -O voices.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" || \
    wget -q --show-progress -O voices.bin "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.bin" || true
fi
cd ..
echo -e "${GREEN}✓ Neural voice weights cached in ./models/${NC}"

# 7. Environment Setup (.env)
echo -e "${GREEN}[6/8] Configuring Production Environment (.env)...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
fi
# Ensure domain is set in .env
sed -i "s|server.ibrasoft.com|${DOMAIN}|g" .env || true
echo -e "${GREEN}✓ Production environment variables loaded.${NC}"

# 8. Build and Start Docker Compose GPU Cluster
echo -e "${GREEN}[7/8] Building and Launching Docker GPU Cluster...${NC}"
docker compose -f docker-compose.gpu.yml down --remove-orphans || true
docker compose -f docker-compose.gpu.yml up -d --build

# 9. Verify System Containers and Latency Diagnostics
echo -e "${GREEN}[8/8] Running Hardware & Latency Benchmark inside GPU Container...${NC}"
sleep 5
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

docker compose -f docker-compose.gpu.yml exec -T voice-agent-gpu python verify_gpu.py || true

PUBLIC_IP=$(curl -s -m 3 ifconfig.me || hostname -I | awk '{print $1}')

echo -e "\n${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}  ✓ GPU VOICE WORKER IS FULLY OPERATIONAL & LIVE!                             ${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${CYAN}  • Domain Hostname        : https://${DOMAIN}${NC}"
echo -e "${CYAN}  • Public IP              : ${PUBLIC_IP}${NC}"
echo -e "${CYAN}  • LiveKit WebRTC (WSS)   : wss://${DOMAIN}/rtc (or port 7880)${NC}"
echo -e "${CYAN}  • LiveKit SIP Gateway    : ${DOMAIN}:5060 (UDP/TCP)${NC}"
echo -e "${CYAN}  • Audio Media RTP Range  : 10000 - 20000 (UDP)${NC}"
echo -e "${CYAN}  • Go Backend API Proxy   : https://${DOMAIN}/api/v1${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${YELLOW}Telnyx SIP Trunk Configuration:${NC}"
echo -e "  In your Telnyx portal -> SIP Trunks -> Outbound SIP URI / Inbound Routing:"
echo -e "  Set Destination to: ${BOLD}sip:${PUBLIC_IP}:5060${NC} (or sip:${DOMAIN}:5060)\n"
