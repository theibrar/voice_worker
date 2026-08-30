# 🚀 GPU Node Deployment Guide: `server.ibrasoft.com`

This guide walks you through setting up and running the **GPU Voice AI Worker Node** on your rented GPU instance (Ubuntu 22.04 / 24.04 with NVIDIA GPU).

---

## 📋 Prerequisites Checklist
1. **Rented GPU Instance**: (e.g. RunPod, Vast.ai, Lambda Labs, AWS EC2 g5, DigitalOcean GPU, Hetzner, etc.) with Ubuntu.
2. **DNS A Record**: Point **`server.ibrasoft.com`** $\to$ **Your GPU Server's Public IP**.
3. **Telnyx Outbound SIP**: Set to **`sip:server.ibrasoft.com:5060`** (or `sip:<YOUR_GPU_PUBLIC_IP>:5060`).

---

## ⚡ 1-Step Automated Setup Command

Once you SSH into your rented GPU server, run:

```bash
# 1. Clone your repository (or upload the voice-worker folder)
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO>/voice-worker

# 2. Make scripts executable
chmod +x setup_gpu.sh start.sh verify_gpu.py

# 3. Run the automated installer (installs Docker, CUDA toolkit, models, firewall, and starts containers)
sudo ./setup_gpu.sh server.ibrasoft.com
```

---

## 🔧 Manual Step-by-Step (If Running Separately)

### Step 1: Configure `.env`
```bash
cp .env.example .env
nano .env
```
*(Add your `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` and verify Telnyx credentials)*

### Step 2: Download Kokoro-82M Weights
```bash
mkdir -p models
wget -O models/kokoro-v0_19.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
wget -O models/voices.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
```

### Step 3: Launch Docker GPU Stack
```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

### Step 4: Verify GPU Hardware & Benchmark
```bash
docker compose -f docker-compose.gpu.yml exec voice-agent-gpu python verify_gpu.py
```

---

## 📊 Telephony & Port Mapping Summary

| Service | Port | Protocol | Usage |
| :--- | :--- | :--- | :--- |
| **SIP Signaling (Telnyx)** | `5060` | UDP / TCP | Inbound & Outbound VoIP Calls |
| **RTP Media Streams** | `10000 - 20000` | UDP | High-speed 24kHz Raw PCM Audio |
| **LiveKit WebRTC Signaling** | `7880` | TCP / WSS | Browser / App WebRTC voice connection |
| **LiveKit STUN / TURN** | `3478`, `5349` | UDP / TCP | NAT traversal |
| **Nginx HTTPS & WSS Gateway** | `80`, `443` | TCP / SSL | `https://server.ibrasoft.com` |

---

## 🔍 Useful Diagnostic Commands

- **Check Running Containers**:
  ```bash
  docker ps
  ```
- **Stream Live Call Audio & Agent Logs**:
  ```bash
  docker compose -f docker-compose.gpu.yml logs -f voice-agent-gpu
  ```
- **Check LiveKit SIP Gateway Status**:
  ```bash
  docker compose -f docker-compose.gpu.yml logs -f livekit-sip
  ```
- **Check GPU VRAM & CUDA Utilization**:
  ```bash
  nvidia-smi -l 1
  ```
