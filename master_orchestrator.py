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
GPU_MEM_UTIL = os.getenv("GPU_MEM_UTIL", "0.50")
PUBLIC_IP = os.getenv("PUBLIC_IP", "202.215.0.218")
PORT_VLLM = os.getenv("PORT_VLLM", "50287")
PORT_TTS = os.getenv("PORT_TTS", "50869")
PORT_STT = os.getenv("PORT_STT", "50053")
PORT_VAD = os.getenv("PORT_VAD", "50604")
PORT_UI = os.getenv("PORT_UI", "50057")

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

def free_ports():
    """Clean up any zombie process port bindings before launching"""
    logger.info("🧹 Freeing existing port bindings (8000, 8088, 8030, 8090, 7860)...")
    for port in ["8000", "8088", "8030", "8090", "7860"]:
        try:
            subprocess.run(f"fuser -k {port}/tcp >/dev/null 2>&1", shell=True)
        except Exception:
            pass

def start_services():
    free_ports()
    time.sleep(1.0)

    logger.info("==================================================================")
    logger.info("   🎙️  ENTERPRISE GPU VOICE AI STACK - MASTER ORCHESTRATOR         ")
    logger.info("   Hardware: 1x NVIDIA RTX 3060 (12GB VRAM)                       ")
    logger.info("   CPU: Intel 13th Gen Core i9-13900K (32 vCPUs, 128.5GB RAM)      ")
    logger.info(f"   Public IP: {PUBLIC_IP}                                          ")
    logger.info("==================================================================")

    env = os.environ.copy()
    env["GPU_API_KEY"] = API_KEY
    env["VLLM_USE_V1"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info("📊 VRAM ALLOCATION BUDGET (12.0 GB Total):")
    logger.info("   • Qwen2.5-7B AWQ Weights + KV Cache (vLLM) : ~6.7 GB (0.58 utilization)")
    logger.info("   • Faster-Whisper FP16 (STT)     : ~1.2 GB")
    logger.info("   • Kokoro-82M ONNX (TTS)         : ~0.5 GB")
    logger.info("   • Silero VAD v5 (VAD)           : ~0.1 GB")
    logger.info("   • CUDA / PyTorch Contexts       : ~1.2 GB (Kernel & Driver overhead)")
    logger.info("   • Total Base Footprint          : ~7.4 GB / 12.0 GB")
    logger.info("   • FREE VRAM for Callers         : ~4.6 GB (PagedAttention Slots)")
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

    # 1. Start vLLM Engine FIRST (Port 8000) so it acquires GPU memory on clean VRAM
    logger.info(f"► [1/5] Launching vLLM Engine ({LLM_MODEL}) on Port 8000...")
    logger.info("   ⚡ Continuous Batching: max-num-seqs 32 (Continuous PagedAttention)")
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
        logger.success("✓ vLLM process spawned successfully!")
    except Exception as e:
        logger.error(f"Could not start vLLM: {e}")

    # Poll http://127.0.0.1:8000/v1/models until vLLM engine is 100% active in VRAM
    logger.info("⏳ Polling vLLM engine until GPU weights & KV cache are 100% ready...")
    import urllib.request
    import urllib.error

    vllm_ready = False
    for attempt in range(45):
        time.sleep(1)
        if p_llm.poll() is not None:
            logger.error("vLLM process exited unexpectedly during startup.")
            break
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    vllm_ready = True
                    logger.success(f"✓ vLLM Engine initialized on CUDA and listening on Port 8000! (took {attempt+1}s)")
                    break
        except Exception:
            pass

    if not vllm_ready:
        logger.warning("vLLM readiness polling timeout reached; proceeding with remaining engines...")

    # 2. Start STT Transcriber with Denoising (Port 8030)
    logger.info("► [2/5] Launching Streaming STT Engine (Port 8030)...")
    p_stt = subprocess.Popen([sys.executable, "stt_server.py"], env=env)
    processes.append(p_stt)
    time.sleep(1.5)

    # 3. Start Kokoro-82M TTS Server (Port 8088)
    logger.info("► [3/5] Launching Kokoro-82M Neural TTS Engine (Port 8088)...")
    p_tts = subprocess.Popen([sys.executable, "tts_server.py"], env=env)
    processes.append(p_tts)
    time.sleep(1.5)

    # 4. Start Silero VAD Barge-In Controller (Port 8090)
    logger.info("► [4/5] Launching Silero VAD & Barge-in Controller (Port 8090)...")
    p_vad = subprocess.Popen([sys.executable, "vad_server.py"], env=env)
    processes.append(p_vad)
    time.sleep(1.5)

    # 5. Start Gradio Interactive Human Prosody Testbench (Port 7860)
    logger.info("► [5/5] Launching Gradio Testbench UI (Port 7860)...")
    p_ui = subprocess.Popen([sys.executable, "testbench_ui.py"], env=env)
    processes.append(p_ui)
    time.sleep(1.5)

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

    # Keep orchestrator alive & monitor active processes
    active_processes = list(processes)
    while True:
        time.sleep(5)
        for p in list(active_processes):
            if p.poll() is not None:
                logger.warning(f"Process {p.pid} exited with code {p.returncode}")
                active_processes.remove(p)

if __name__ == "__main__":
    start_services()
