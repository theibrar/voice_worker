# 🎙️ Apex Enterprise Voice AI - GPU Worker Node
### Hardware: 1x NVIDIA RTX 3090 (24GB VRAM) | Public IP: `212.93.107.107`

This repository turns your **Vast.ai RTX 3090 GPU instance** into an enterprise-grade, human-realistic Voice AI cluster powering:
1. **vLLM Engine (Port 8000)**: Qwen 2.5 7B Instruct with prefix caching and native tool calling.
2. **Kokoro-82M Streaming Neural TTS (Port 8088)**: Chunked PCM streaming (<150ms TTFA) with emotional prosody tags (`[empathy]`, `[cheerful]`, `[urgent]`).
3. **High-Speed Streaming STT (Port 8030)**: Faster-Whisper on CUDA with PSTN high-pass audio denoising and speculative entity pre-fetching.
4. **Silero VAD Barge-In Engine (Port 8090)**: Real-time -20dB speech detection to cut audio when human interrupts.
5. **Gradio Audio Testbench (Port 7860)**: Direct browser UI to test mic, prosody, and measure sub-300ms latency without making a phone call.

---

## 🚀 Quick Setup (1-Click)

### Step 1: SSH into your Vast.ai GPU instance
```bash
ssh -p 41103 root@212.93.107.107
```

### Step 2: Clone or Pull this repository
```bash
git clone https://github.com/theibrar/voice_worker.git ~/voice_worker || (cd ~/voice_worker && git pull)
cd ~/voice_worker
```

### Step 3: Run the Automated Installer
```bash
chmod +x setup.sh
bash setup.sh
```

The script will:
* Verify your NVIDIA RTX 3090 GPU and CUDA drivers.
* Install all required audio DSP libraries and Python AI packages.
* Prompt you for your preferred secret API key (default: `sk-ibrasoft-gpu-voice`).
* Cache the Kokoro neural voice models.
* Launch all 5 engines in a persistent background `tmux` session.

---

## 🌐 Your Live Production Endpoints

| Service | Internal Port | Vast.ai Public Mapped Port | Public Base URL |
| :--- | :--- | :--- | :--- |
| **Gradio Web Audio Testbench** | `7860` | **`41064`** | **`http://212.93.107.107:41064`** |
| **vLLM OpenAI-Compatible API** | `8000` | **`41091`** | **`http://212.93.107.107:41091/v1`** |
| **Kokoro Neural Streaming TTS** | `8088` | **`41438`** | **`http://212.93.107.107:41438`** |
| **Faster STT with Denoising** | `8030` | **`41182`** | **`http://212.93.107.107:41182`** |
| **Silero VAD Barge-In Engine** | `8090` | **`41423`** | **`http://212.93.107.107:41423`** |

---

## 🛠️ Adding to Contabo Super Admin

Log in to your Contabo platform dashboard at **`https://agents.ibrasoft.com/super-admin/engines`** (or `http://localhost:3000/super-admin/engines`):

### 1. Register Private LLM
* Click **"+ Register Custom Engine"**
* **Category**: `LLM Reasoning`
* **Name**: `Qwen-2.5-7B Private GPU`
* **Provider**: `OpenAI-Compatible vLLM`
* **Model Identifier**: `Qwen/Qwen2.5-7B-Instruct`
* **Base URL**: `http://212.93.107.107:41091/v1`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `60 ms`

### 2. Register Private Neural TTS
* Click **"+ Register Custom Engine"**
* **Category**: `TTS Neural Audio`
* **Name**: `Kokoro-82M Streaming GPU`
* **Provider**: `Kokoro Neural`
* **Model Identifier**: `kokoro-v0_19`
* **Base URL**: `http://212.93.107.107:41438`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `30 ms`

### 3. Register Private STT
* Click **"+ Register Custom Engine"**
* **Category**: `STT Transcription`
* **Name**: `Fast-Whisper Denoised GPU`
* **Provider**: `Whisper / Parakeet`
* **Model Identifier**: `distil-large-v3`
* **Base URL**: `http://212.93.107.107:41182`
* **API Key**: `sk-ibrasoft-gpu-voice`
* **Estimated Latency**: `65 ms`

---

## 🧪 Monitoring & Testing

* **View live engine logs on GPU**:
  ```bash
  tmux attach -t voice-worker
  ```
  *(Press `Ctrl+B` then `D` to detach without stopping services)*

* **Test Live Mic in Browser**:
  Open `http://212.93.107.107:41064` in your web browser to test talking into your microphone and measuring real-time turn latency!
