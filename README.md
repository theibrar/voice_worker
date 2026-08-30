# Enterprise Voice AI Worker (GPU / Local Pipeline)
### *NVIDIA FastConformer Parakeet STT + Kokoro-82M Neural Voice Engine + LiveKit WebRTC & SIP*

---

## 🚀 Quick Start: 2 Ways to Run

### Option 1: Local Testing on Windows / Mac / Linux (Zero GPU Required)

To test the entire audio pipeline, autonomous tools (RAG, SMS, Calendar), and PostgreSQL credit deduction locally on your PC:

1. **Install Python Dependencies:**
   ```bash
   cd voice-worker
   pip install -r requirements.txt
   ```

2. **Run Interactive Simulator:**
   ```bash
   python test_simulator.py
   ```
   * It will handshake with your local Go Backend on port `8080`.
   * Type or speak your questions (e.g. *"What is your warranty?"* or *"Can you text me your brochure?"*).
   * Displays exact turn latencies in milliseconds.
   * On exit, updates call duration and deducts 1 credit per minute from PostgreSQL.

---

### Option 2: Production Deployment on Ubuntu GPU Server

To deploy on a remote Ubuntu GPU server (Ubuntu 22.04 / 24.04 LTS with NVIDIA CUDA):

1. **Upload the `voice-worker/` folder to your GPU server:**
   ```bash
   scp -r voice-worker user@your-gpu-server-ip:/home/user/
   ```

2. **SSH into the GPU server and run the automated installer:**
   ```bash
   cd /home/user/voice-worker
   sudo bash setup_gpu.sh
   ```

   **What `setup_gpu.sh` does automatically:**
   * ✅ Verifies and installs NVIDIA Drivers & CUDA 12.4
   * ✅ Installs Docker & NVIDIA Container Toolkit (`nvidia-ctk`)
   * ✅ Configures UFW Firewall (`5060/udp` SIP, `7880/tcp` LiveKit, `10000-20000/udp` RTP Media)
   * ✅ Pre-caches Kokoro-82M Neural Weights (`af_heart`, `af_bella`, `am_adam`, etc.)
   * ✅ Launches the production `docker-compose.gpu.yml` stack with hardware GPU reservations.

---

## 📁 Directory Structure

```
voice-worker/
├── setup_gpu.sh              # One-click automated Ubuntu GPU installer
├── agent.py                  # Production LiveKit Voice Agent Worker
├── kokoro_tts_engine.py      # Kokoro-82M Neural TTS Engine (Sub-100ms)
├── parakeet_stt_engine.py    # NVIDIA FastConformer Streaming STT (Sub-80ms)
├── turn_detector.py          # Dual-Stage Silero VAD & Interruption Handler
├── tools/                    # Autonomous Mid-Call Tools
│   ├── calendar_tool.py      # Live Google Calendar Slot Booking
│   ├── sms_tool.py           # Real-Time SMS Dispatch during Call
│   ├── transfer_tool.py      # SIP REFER Warm & Cold Transfers
│   └── rag_tool.py           # pgvector Semantic Vector Search
├── livekit.yaml              # LiveKit Server configuration
├── livekit-sip.yaml          # Telnyx SIP Inbound/Outbound Trunk mapping
├── docker-compose.gpu.yml    # Production Docker GPU stack
├── docker-compose.local.yml  # Local Docker dev stack
├── Dockerfile.gpu            # CUDA 12.4 PyTorch container
├── Dockerfile.local          # Lightweight Python container
├── requirements.txt          # Python package requirements
├── .env.example              # Environment variables template
└── test_simulator.py         # Local CLI call simulator & latency benchmark
```

---

## 📞 Pointing Telnyx SIP Trunk to Your GPU Server

1. Open your **Telnyx Mission Control Portal**.
2. Go to **SIP Connections** ➔ **Add SIP Connection**.
3. Set **SIP Connection Type** to `FQDN / IP`.
4. Enter your GPU Server Public IP and Port `5060` (UDP).
5. Assign your purchased Inbound DID (e.g. `+14156390491`) to this SIP connection.
6. When someone dials your number, Telnyx immediately bridges audio to LiveKit SIP, your Agent answers in sub-250ms, and your Go backend deducts 1 credit per minute.
