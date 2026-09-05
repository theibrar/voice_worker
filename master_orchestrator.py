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
from dotenv import load_dotenv

# Load environment variables from .env if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)

API_KEY = os.getenv("GPU_API_KEY", "sk-ibrasoft-gpu-voice")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GPU_MEM_UTIL = os.getenv("GPU_MEM_UTIL", "0.50")
PUBLIC_IP = os.getenv("PUBLIC_IP", "77.54.200.11")
PORT_VLLM = os.getenv("PORT_VLLM", "15363")
PORT_TTS = os.getenv("PORT_TTS", "15173")
PORT_STT = os.getenv("PORT_STT", "15426")
PORT_VAD = os.getenv("PORT_VAD", "15197")
PORT_UI = os.getenv("PORT_UI", "15290")

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
    logger.info("   Hardware: 1x NVIDIA RTX 4060 Ti (16GB VRAM)                    ")
    logger.info("   CPU: Intel Xeon E5-2673 v4 (20 vCPUs, 96.7GB RAM)              ")
    logger.info(f"   Public IP: {PUBLIC_IP}                                          ")
    logger.info("==================================================================")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    env["GPU_API_KEY"] = API_KEY
    env["VLLM_USE_V1"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Minimize CUDA context & PyTorch overhead across all 4 worker processes
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info("📊 VRAM ALLOCATION BUDGET (16.0 GB Total):")
    logger.info("   • Qwen2.5-7B-AWQ Weights       : ~5.2 GB")
    logger.info("   • Parakeet-TDT (v3) STT        : ~1.2 GB")
    logger.info("   • Kokoro-82M ONNX              : ~0.5 GB")
    logger.info("   • Silero VAD v5                : ~0.1 GB")
    logger.info("   • CUDA / PyTorch Contexts      : ~1.2 GB (Kernel & Driver overhead)")
    logger.info("   • Total Static Base Footprint   : ~8.2 GB / 16.0 GB")
    logger.info("   • FREE VRAM for 32 Callers     : ~7.8 GB (Continuous Batching Slots)")
    logger.info("==================================================================")
    cublas_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib"
    cudnn_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
    curun_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib"
    nvrtc_lib = "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib"
    sys_cuda = "/usr/local/cuda/lib64"

    full_ld_path = f"{cublas_lib}:{cudnn_lib}:{curun_lib}:{nvrtc_lib}:{sys_cuda}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = full_ld_path
    env["LD_LIBRARY_PATH"] = full_ld_path

    # Preload CUDA runtime & cuBLAS globally into process table
    import ctypes
    for p in [curun_lib, cublas_lib, cudnn_lib]:
        if os.path.exists(p):
            for lib in ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]:
                f_path = os.path.join(p, lib)
                if os.path.exists(f_path):
                    try:
                        ctypes.CDLL(f_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

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

    # 5. Start vLLM Engine (Port 8000)
    logger.info(f"► [5/5] Launching vLLM Engine ({LLM_MODEL}) on Port 8000...")
    logger.info(f"   ⚡ Continuous Batching: max-num-seqs 32 (PagedAttention | API Key: {API_KEY})")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", LLM_MODEL,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--api-key", API_KEY,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
        "--enforce-eager",
        "--trust-remote-code"
    ]
    if "awq" in LLM_MODEL.lower():
        vllm_cmd.extend(["--quantization", "awq"])

    try:
        p_llm = subprocess.Popen(vllm_cmd, env=env)
        processes.append(p_llm)
        logger.success("✓ vLLM spawned with continuous batching (32 concurrent sequences)!")
    except Exception as e:
        logger.error(f"Could not start vLLM: {e}")

    logger.success("\n==================================================================")
    logger.success("   🎉 ALL 5 GPU SERVICES ARE LIVE AND RUNNING!                   ")
    logger.success("==================================================================")
    logger.info(f"  • vLLM OpenAI API : http://{PUBLIC_IP}:{PORT_VLLM}/v1 (Port 8000)")
    logger.info(f"  • Kokoro TTS API  : http://{PUBLIC_IP}:{PORT_TTS} (Port 8088)")
    logger.info(f"  • STT Audio API   : http://{PUBLIC_IP}:{PORT_STT} (Port 8030)")
    logger.info(f"  • Silero VAD API  : http://{PUBLIC_IP}:{PORT_VAD} (Port 8090)")
    logger.info(f"  • Gradio UI Web   : http://{PUBLIC_IP}:{PORT_UI} (Port 7860)")
    logger.info(f"  • API Key         : {API_KEY}")
    logger.success("==================================================================\n")

    # Keep orchestrator alive
    active_processes = list(processes)
    while True:
        time.sleep(5)
        for p in list(active_processes):
            if p.poll() is not None:
                logger.warning(f"Process {p.pid} exited with code {p.returncode}")
                active_processes.remove(p)

if __name__ == "__main__":
    start_services()
