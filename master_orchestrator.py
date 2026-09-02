"""
Enterprise Voice AI GPU Master Orchestrator
Spawns and supervises all 5 specialized GPU AI engines:
1. Port 8000: vLLM OpenAI-Compatible LLM (Qwen/Qwen2.5-7B-Instruct)
2. Port 8088: Kokoro-82M Streaming Neural TTS Server
3. Port 8030: Fast Streaming STT Server with Audio Denoising
4. Port 8090: Silero VAD & Barge-In Controller
5. Port 7860: Gradio Interactive Audio Testbench
"""

import os
import sys
import time
import signal
import subprocess
from loguru import logger

API_KEY = os.getenv("GPU_API_KEY", "sk-ibrasoft-gpu-voice")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GPU_MEM_UTIL = os.getenv("GPU_MEM_UTIL", "0.65")

processes = []

def signal_handler(sig, frame):
    logger.warning("Stopping all GPU voice worker engines...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_services():
    logger.info("==================================================================")
    logger.info("   🎙️  ENTERPRISE GPU VOICE AI STACK - MASTER ORCHESTRATOR         ")
    logger.info("   Hardware: 1x NVIDIA RTX 3060 (12GB VRAM) | AMD EPYC 7502P       ")
    logger.info("   Public IP: 173.185.79.174                                      ")
    logger.info("==================================================================")

    env = os.environ.copy()
    env["GPU_API_KEY"] = API_KEY
    env["VLLM_USE_V1"] = "0"
    cublas_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib"
    cudnn_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
    curun_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib"
    nvrtc_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib"
    sys_cuda = "/usr/local/cuda/lib64"

    full_ld_path = f"{cublas_lib}:{cudnn_lib}:{curun_lib}:{nvrtc_lib}:{sys_cuda}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = full_ld_path
    env["LD_LIBRARY_PATH"] = full_ld_path

    # 1. Start Kokoro-82M TTS Server (Port 8088)
    logger.info("► [1/5] Launching Kokoro-82M Neural TTS Engine (Port 8088)...")
    p_tts = subprocess.Popen([sys.executable, "tts_server.py"], env=env)
    processes.append(p_tts)
    time.sleep(1.5)

    # 2. Start STT Transcriber with Denoising (Port 8030)
    logger.info("► [2/5] Launching Streaming STT Engine (Port 8030)...")
    p_stt = subprocess.Popen([sys.executable, "stt_server.py"], env=env)
    processes.append(p_stt)
    time.sleep(1.5)

    # 3. Start Silero VAD Barge-In Controller (Port 8090)
    logger.info("► [3/5] Launching Silero VAD & Barge-in Controller (Port 8090)...")
    p_vad = subprocess.Popen([sys.executable, "vad_server.py"], env=env)
    processes.append(p_vad)
    time.sleep(1.5)

    # 4. Start Gradio Interactive Human Prosody Testbench (Port 7860)
    logger.info("► [4/5] Launching Gradio Testbench UI (Port 7860)...")
    p_ui = subprocess.Popen([sys.executable, "testbench_ui.py"], env=env)
    processes.append(p_ui)
    time.sleep(1.5)

    # 5. Start vLLM OpenAI-compatible Engine (Port 8000)
    logger.info(f"► [5/5] Launching vLLM Engine ({LLM_MODEL}) on Port 8000...")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", LLM_MODEL,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--api-key", API_KEY,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-model-len", "4096",
        "--quantization", "awq",
        "--enforce-eager",
        "--trust-remote-code"
    ]
    
    try:
        p_vllm = subprocess.Popen(vllm_cmd, env=env)
        processes.append(p_vllm)
    except Exception as e:
        logger.error(f"Could not start vLLM directly: {e}")

    logger.success("\n==================================================================")
    logger.success("   🎉 ALL 5 GPU SERVICES ARE LIVE AND RUNNING!                   ")
    logger.success("==================================================================")
    logger.info(f"  • vLLM OpenAI API : http://173.185.79.174:46409/v1 (Port 8000)")
    logger.info(f"  • Kokoro TTS API  : http://173.185.79.174:47830 (Port 8088)")
    logger.info(f"  • STT Audio API   : http://173.185.79.174:46819 (Port 8030)")
    logger.info(f"  • Silero VAD API  : http://173.185.79.174:49760 (Port 8090)")
    logger.info(f"  • Gradio UI Web   : http://173.185.79.174:47761 (Port 7860)")
    logger.info(f"  • API Key         : {API_KEY}")
    logger.success("==================================================================\n")

    # Keep orchestrator alive
    while True:
        time.sleep(5)
        for p in processes:
            if p.poll() is not None:
                logger.warning(f"Process {p.pid} exited with code {p.returncode}")

if __name__ == "__main__":
    start_services()
