# 🚀 2-Minute GPU Live Deployment Guide (`server.ibrasoft.com`)

This guide gives you the exact, automated steps to deploy your Voice AI worker on your rented Ubuntu GPU server without wasting billable hours.

---

## 📋 Pre-Flight Checklist
1. **Rented Server:** Ubuntu 22.04 LTS or 24.04 LTS with an NVIDIA GPU (RTX 3090, RTX 4090, A4000, A5000, A10G, L4, or A100).
2. **DNS Record:** In your domain DNS manager (e.g. Cloudflare / Namecheap / GoDaddy), point:
   ```
   A Record: server.ibrasoft.com  --->  YOUR_GPU_SERVER_PUBLIC_IP
   ```

---

## ⚡ 1-Click Automated Setup

### Step 1: Upload the `voice-worker` directory to your GPU server
From your local terminal / command prompt:
```bash
scp -r voice-worker user@YOUR_GPU_SERVER_IP:/home/user/
```

### Step 2: SSH into your server & Run the Installer
```bash
ssh user@YOUR_GPU_SERVER_IP
cd /home/user/voice-worker
sudo bash setup_gpu.sh server.ibrasoft.com
```

The script will automatically:
* ✅ Verify NVIDIA GPU drivers and configure Docker NVIDIA runtime (`nvidia-ctk`).
* ✅ Open SIP (`5060`), WebRTC (`7880`), TURN (`3478`), and RTP (`10000-20000`) firewall ports.
* ✅ Generate SSL certificates for `server.ibrasoft.com`.
* ✅ Download & verify Kokoro-82M ONNX model weights and neural voices.
* ✅ Launch the full LiveKit SIP + Parakeet STT + Kokoro-82M TTS container stack.

---

## 📞 Connect with Telnyx VoIP SIP Trunk

1. Open your **Telnyx Mission Control Portal** (`portal.telnyx.com`).
2. Go to **SIP Trunks** $\to$ Select or Create a SIP Connection.
3. Under **Inbound Settings**:
   * **Routing Type:** FQDN / IP Address
   * **Destination SIP URI:** `sip:server.ibrasoft.com:5060` (or `sip:YOUR_GPU_SERVER_IP:5060`)
4. Under **Phone Numbers**:
   * Assign your phone number (e.g. `+14156390491`) to this SIP Connection.

---

## 🧪 Real-Time Latency Benchmark & Diagnostics

To run an instant 5-second diagnostic benchmark:
```bash
docker exec -it voice-agent-gpu python verify_gpu.py
```
This tests:
* CUDA hardware acceleration
* ONNX Runtime GPU provider
* Kokoro-82M First-Byte Latency (sub-40ms on GPU)
* LiveKit WebRTC & SIP listener health

---

## 📊 Container Management Commands
```bash
# View live real-time conversational logs
docker logs -f voice-agent-gpu

# View LiveKit SIP call logs
docker logs -f livekit-sip

# Restart the GPU cluster
docker compose -f docker-compose.gpu.yml restart

# Stop the stack
docker compose -f docker-compose.gpu.yml down
```
