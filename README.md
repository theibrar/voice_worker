# 🎙️ Apex Enterprise Voice AI - GPU Worker Node
### Hardware: 1x NVIDIA RTX 5060 Ti (16GB VRAM) | Intel Xeon E5-2673 v4 (40 vCPUs, 96.5GB RAM)
### Public IP: `77.54.200.11` | Instance ID: `49995859`

This repository turns your **Vast.ai GPU instance** into an enterprise-grade, human-realistic Voice AI cluster powering:
1. **vLLM Engine (Port 8000)**: Qwen 2.5 7B Instruct AWQ with continuous batching, prefix caching, and conversational filler prompting.
2. **Kokoro-82M Streaming Neural TTS (Port 8088)**: Full 54-voice multi-language pack, dynamic voice blending, punctuation-driven prosody, and sub-40ms first-chunk streaming.
3. **NVIDIA Parakeet-TDT Streaming STT (Port 8030)**: FastConformer RNN-T / TDT on CUDA (with safe CPU fallback) with PSTN denoising and speculative entity extraction.
4. **Silero VAD Barge-In Engine (Port 8090)**: Real-time 32ms frame speech detection to cut audio when human interrupts.
5. **Gradio Audio Testbench (Port 7860)**: Direct browser UI to test mic, prosody, dynamic voice blending, and measure sub-200ms latency.

---

## 🚀 Quick Setup (1-Click)

### Step 1: SSH into your Vast.ai GPU instance
```bash
ssh -p 15475 root@77.54.200.11
```

### Step 2: Clone or Pull this repository
```bash
git clone https://github.com/thewh1teagle/voice_worker.git ~/voice_worker || (cd ~/voice_worker && git pull)
cd ~/voice_worker
```

### Step 3: Run the Automated Installer
```bash
chmod +x setup.sh
./setup.sh
```

The script will:
* Verify your NVIDIA RTX 5060 Ti GPU and CUDA drivers.
* Install system dependencies (`libcublas-12-0`, `ffmpeg`, `sox`).
* Install Python packages (`nvidia-cublas-cu12`, `vllm`, `faster-whisper`, `kokoro-onnx`).
* Download and verify clean `kokoro-v0_19.onnx` and `voices.bin`.
* Launch all 5 engines in a persistent background `tmux` session.

---

## 🌐 Live Production Endpoints

| Service | Container Port | Vast.ai Public Mapped Port | Public Base URL |
| :--- | :--- | :--- | :--- |
| **Gradio Web Audio Testbench** | `7860` | **`15238`** | **`http://77.54.200.11:15238`** |
| **vLLM OpenAI-Compatible API** | `8000` | **`15460`** | **`http://77.54.200.11:15460/v1`** |
| **Kokoro Neural Streaming TTS** | `8088` | **`15188`** | **`http://77.54.200.11:15188`** |
| **NVIDIA Parakeet-TDT STT** | `8030` | **`15490`** | **`http://77.54.200.11:15490`** |
| **Silero VAD Barge-In Engine** | `8090` | **`15089`** | **`http://77.54.200.11:15089`** |

---

## 🛠️ Super Admin Engine Registration Details

Log in to your platform dashboard at **`/super-admin/engines`**:

### 1. Register Private LLM
* Click **"+ Register Custom Engine"**
* **Category**: `LLM Reasoning`
* **Name**: `Qwen-2.5-7B Private GPU`
* **Provider**: `OpenAI-Compatible vLLM`
* **Model Identifier**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
* **Base URL**: `http://77.54.200.11:15460/v1`
* **API Key**: `<YOUR_GPU_API_KEY>`
* **Estimated Latency**: `45 ms`

### 2. Register Private Neural TTS
* Click **"+ Register Custom Engine"**
* **Category**: `TTS Neural Audio`
* **Name**: `Kokoro-82M Streaming GPU (54 Voices & Blending)`
* **Provider**: `Kokoro Neural`
* **Model Identifier**: `kokoro-v1.0`
* **Base URL**: `http://77.54.200.11:15188`
* **API Key**: `<YOUR_GPU_API_KEY>`
* **Estimated Latency**: `35 ms`
* **Supported Voices**: Full 54-Voice Pack + Dynamic Blends (e.g., `af_bella:0.82,am_michael:0.18`)

### 3. Register Private STT
* Click **"+ Register Custom Engine"**
* **Category**: `STT Transcription`
* **Name**: `NVIDIA Parakeet-TDT GPU STT`
* **Provider**: `NVIDIA Parakeet-TDT / FastConformer`
* **Model Identifier**: `nvidia/parakeet-tdt-1.1b`
* **Base URL**: `http://77.54.200.11:15490`
* **API Key**: `<YOUR_GPU_API_KEY>`
* **Estimated Latency**: `50 ms`

### 4. Register Private VAD / Telephony Interrupter
* Click **"+ Register Custom Engine"**
* **Category**: `VAD Interruption`
* **Name**: `Silero VAD v5 Neural`
* **Provider**: `Silero VAD`
* **Model Identifier**: `silero-v5`
* **Base URL**: `http://77.54.200.11:15089`
* **WebSocket URL**: `ws://77.54.200.11:15089/vad/stream`
* **API Key**: `<YOUR_GPU_API_KEY>`
* **Estimated Latency**: `15 ms`

---

## 🧪 Monitoring & Testing

* **View live engine logs on GPU**:
  ```bash
  tmux attach -t voice-worker
  ```
  *(Press `Ctrl+B` then `D` to detach without stopping services)*

* **Test Live Mic in Browser**:
  Open `http://77.54.200.11:15238` in your web browser to test talking into your microphone and measuring real-time turn latency!
