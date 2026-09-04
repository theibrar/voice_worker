# 🎙️ Apex Enterprise Voice AI - GPU Worker Node
### Hardware: 1x NVIDIA RTX 5060 Ti (16GB VRAM) | Intel Xeon E5-2673 v4 (40 vCPUs, 96.5GB RAM)
### Public IP: `184.144.154.180` | Instance ID: `49888185`

This repository turns your **Vast.ai GPU instance** into an enterprise-grade, human-realistic Voice AI cluster powering:
1. **vLLM Engine (Port 8000)**: Qwen 2.5 7B Instruct AWQ with continuous batching, prefix caching, and OpenAI compatibility.
2. **Kokoro-82M Streaming Neural TTS (Port 8088)**: Chunked PCM streaming (<50ms TTFA) with emotional prosody tags (`[empathy]`, `[cheerful]`, `[urgent]`).
3. **High-Speed Streaming STT (Port 8030)**: Faster-Whisper distil-large-v3 on CUDA (float16) with PSTN denoising and speculative entity extraction.
4. **Silero VAD Barge-In Engine (Port 8090)**: Real-time 32ms frame speech detection to cut audio when human interrupts.
5. **Gradio Audio Testbench (Port 7860)**: Direct browser UI to test mic, prosody, and measure sub-300ms latency.

---

## 🚀 Quick Setup (1-Click)

### Step 1: SSH into your Vast.ai GPU instance
```bash
ssh -p 56082 root@184.144.154.180
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
| **Gradio Web Audio Testbench** | `7860` | **`56081`** | **`http://184.144.154.180:56081`** |
| **vLLM OpenAI-Compatible API** | `8000` | **`56137`** | **`http://184.144.154.180:56137/v1`** |
| **Kokoro Neural Streaming TTS** | `8088` | **`56209`** | **`http://184.144.154.180:56209`** |
| **Faster STT with Denoising** | `8030` | **`56546`** | **`http://184.144.154.180:56546`** |
| **Silero VAD Barge-In Engine** | `8090` | **`56756`** | **`http://184.144.154.180:56756`** |

---

## 🛠️ Super Admin Engine Registration Details

Log in to your platform dashboard at **`/super-admin/engines`**:

### 1. Register Private LLM
* Click **"+ Register Custom Engine"**
* **Category**: `LLM Reasoning`
* **Name**: `Qwen-2.5-7B Private GPU`
* **Provider**: `OpenAI-Compatible vLLM`
* **Model Identifier**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
* **Base URL**: `http://184.144.154.180:56137/v1`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `45 ms`

### 2. Register Private Neural TTS
* Click **"+ Register Custom Engine"**
* **Category**: `TTS Neural Audio`
* **Name**: `Kokoro-82M Streaming GPU`
* **Provider**: `Kokoro Neural`
* **Model Identifier**: `kokoro-v0_19`
* **Base URL**: `http://184.144.154.180:56209`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `35 ms`
* **Supported Voices**: `af_bella`, `am_michael`, `am_adam`, `af_sarah`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

### 3. Register Private STT
* Click **"+ Register Custom Engine"**
* **Category**: `STT Transcription`
* **Name**: `Fast-Whisper Denoised GPU`
* **Provider**: `Whisper / Faster-Whisper`
* **Model Identifier**: `distil-large-v3`
* **Base URL**: `http://184.144.154.180:56546`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `180 ms`

### 4. Register Private VAD / Telephony Interrupter
* Click **"+ Register Custom Engine"**
* **Category**: `VAD Interruption`
* **Name**: `Silero VAD v5 Neural`
* **Provider**: `Silero VAD`
* **Model Identifier**: `silero-v5`
* **Base URL**: `http://184.144.154.180:56756`
* **WebSocket URL**: `ws://184.144.154.180:56756/vad/stream`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `15 ms`

---

## 🧪 Monitoring & Testing

* **View live engine logs on GPU**:
  ```bash
  tmux attach -t voice-worker
  ```
  *(Press `Ctrl+B` then `D` to detach without stopping services)*

* **Test Live Mic in Browser**:
  Open `http://184.144.154.180:56081` in your web browser to test talking into your microphone and measuring real-time turn latency!
