# 🎙️ Apex Enterprise Voice AI - GPU Worker Node
### Hardware: 1x NVIDIA RTX 4060 Ti (16GB VRAM) | Intel Xeon E5-2673 v4 (20 vCPUs, 96.7GB RAM)
### Public IP: `77.54.200.11` | Instance ID: `49995859`

This repository turns your **Vast.ai GPU instance** into an enterprise-grade, human-realistic Voice AI cluster powering:
1. **vLLM Engine (Port 8000)**: Qwen 2.5 7B Instruct AWQ with continuous batching, prefix caching, and OpenAI compatibility.
2. **Kokoro-82M Streaming Neural TTS (Port 8088)**: Chunked PCM streaming (<50ms TTFA) with 54 voices and emotional prosody tags (`[empathy]`, `[cheerful]`, `[urgent]`).
3. **High-Speed Streaming STT (Port 8030)**: NVIDIA Parakeet-TDT (v3) ASR Engine with PSTN denoising and speculative entity extraction.
4. **Silero VAD Barge-In Engine (Port 8090)**: Real-time 32ms frame speech detection to cut audio when human interrupts.
5. **Gradio Audio Testbench (Port 7860)**: Direct browser UI to test mic, prosody, and measure sub-300ms latency.

---

## 🚀 Quick Setup (1-Click)

### Step 1: SSH into your Vast.ai GPU instance
```bash
ssh -p 15148 root@77.54.200.11
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
* Verify your NVIDIA RTX 4060 Ti GPU and CUDA drivers.
* Prompt for your Master Global API Key (or default to `sk-ibrasoft-gpu-voice`).
* Install system & Python dependencies (`libcublas-12-0`, `vllm`, `faster-whisper`, `nemo_toolkit`, `kokoro-onnx`).
* Download multi-language `kokoro-v1.0.onnx` and `voices-v1.0.bin` (54 voices).
* Launch all 5 engines in a persistent background `tmux` session (`tmux attach -t voice-worker`).

---

## 🌐 Live Production Endpoints

| Service | Container Port | Vast.ai Public Mapped Port | Public Base URL |
| :--- | :--- | :--- | :--- |
| **Gradio Web Audio Testbench** | `7860` | **`15290`** | **`http://77.54.200.11:15290`** |
| **vLLM OpenAI-Compatible API** | `8000` | **`15363`** | **`http://77.54.200.11:15363/v1`** |
| **Kokoro Neural Streaming TTS** | `8088` | **`15173`** | **`http://77.54.200.11:15173`** |
| **Faster STT with Denoising** | `8030` | **`15426`** | **`http://77.54.200.11:15426`** |
| **Silero VAD Barge-In Engine** | `8090` | **`15197`** | **`http://77.54.200.11:15197`** |

---

## 🛠️ Super Admin Engine Registration Details

Log in to your platform dashboard at **`/super-admin/engines`**:

### 1. Register Private LLM
* Click **"+ Register Custom Engine"**
* **Category**: `LLM Reasoning`
* **Name**: `Qwen-2.5-7B Private GPU`
* **Provider**: `OpenAI-Compatible vLLM`
* **Model Identifier**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
* **Base URL**: `http://77.54.200.11:15363/v1`
* **API Key**: `sk-ibrasoft-gpu-voice` (or your custom API key)
* **Estimated Latency**: `45 ms`

### 2. Register Private Neural TTS
* Click **"+ Register Custom Engine"**
* **Category**: `TTS Neural Audio`
* **Name**: `Kokoro-82M Streaming GPU (54 Voices)`
* **Provider**: `Kokoro Neural`
* **Model Identifier**: `kokoro-v1.0`
* **Base URL**: `http://77.54.200.11:15173`
* **API Key**: `sk-ibrasoft-gpu-voice` (or your custom API key)
* **Estimated Latency**: `35 ms`
* **Supported Voices**: 54 multi-language voices (`af_heart`, `af_bella`, `am_michael`, `am_adam`, `bf_emma`, `es_dora`, `ff_siwis`, `it_paola`, `ja_alpha`, `zh_xiaobei`, `hi_hindi`)

### 3. Register Private STT
* Click **"+ Register Custom Engine"**
* **Category**: `STT Transcription`
* **Name**: `NVIDIA Parakeet-TDT (v3) GPU`
* **Provider**: `NVIDIA Parakeet / NeMo ASR`
* **Model Identifier**: `nvidia/parakeet-tdt-1.1b`
* **Base URL**: `http://77.54.200.11:15426`
* **API Key**: `sk-ibrasoft-gpu-voice` (or your custom API key)
* **Estimated Latency**: `65 ms`

### 4. Register Private VAD / Telephony Interrupter
* Click **"+ Register Custom Engine"**
* **Category**: `VAD Interruption`
* **Name**: `Silero VAD v5 Neural`
* **Provider**: `Silero VAD`
* **Model Identifier**: `silero-v5`
* **Base URL**: `http://77.54.200.11:15197`
* **WebSocket URL**: `ws://77.54.200.11:15197/vad/stream`
* **API Key**: `sk-ibrasoft-gpu-voice` (or your custom API key)
* **Estimated Latency**: `15 ms`
