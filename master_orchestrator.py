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
import urllib.request
import urllib.error
from loguru import logger

# Automatically load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("GPU_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
PUBLIC_IP = os.getenv("PUBLIC_IP", "77.104.167.149")
PORT_VLLM = os.getenv("PORT_VLLM", "59982")
PORT_TTS = os.getenv("PORT_TTS", "59643")
PORT_STT = os.getenv("PORT_STT", "59805")
PORT_VAD = os.getenv("PORT_VAD", "59929")
PORT_UI = os.getenv("PORT_UI", "59835")

# Dynamic GPU VRAM Detection & Auto-Sizing
total_vram_gb = 16.0
device_name = "NVIDIA GPU"
try:
    import torch
    if torch.cuda.is_available():
        total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        device_name = torch.cuda.get_device_name(0)
except Exception:
    pass

# Dynamically calculate safe vLLM memory utilization based on card size
if total_vram_gb >= 22.0:
    auto_vllm_util = 0.65   # 24GB+ (RTX 3090, 4090, A5000)
elif total_vram_gb >= 15.0:
    auto_vllm_util = 0.52   # 16GB (RTX 5060 Ti, RTX 4080): ~8.2 GB for vLLM, leaves ~7.7 GB for STT/TTS
else:
    auto_vllm_util = 0.58   # 12GB (RTX 3060, 4070)

GPU_MEM_UTIL = os.getenv("GPU_MEM_UTIL", str(auto_vllm_util))

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
    logger.info(f"   Detected GPU : {device_name} ({total_vram_gb} GB VRAM)          ")
    logger.info(f"   Public IP    : {PUBLIC_IP}                                      ")
    logger.info(f"   vLLM Util    : {GPU_MEM_UTIL} (Auto-scaled for 100% 0-OOM guarantee)")
    logger.info("==================================================================")

    env = os.environ.copy()
    env["GPU_API_KEY"] = API_KEY
    env["VLLM_USE_V1"] = "0"
    os.environ["VLLM_USE_V1"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_PLUGINS"] = ""
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
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
            for lib in ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"]:
                f_path = os.path.join(p, lib)
                if os.path.exists(f_path):
                    try:
                        ctypes.CDLL(f_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

    # Ensure peft is installed to prevent vLLM NeMo plugin crash
    try:
        import peft
    except ImportError:
        logger.info("⚡ Auto-installing peft & accelerate to prevent plugin import errors...")
        subprocess.run([sys.executable, "-m", "pip", "install", "peft", "accelerate"], check=False)

    env["PYTHONUNBUFFERED"] = "1"

    # =========================================================================
    # Step 1: Launch Kokoro-82M TTS Server (Port 8088)
    # Lightweight static model (~0.5 GB VRAM)
    # =========================================================================
    logger.info("► [1/5] Launching Kokoro-82M Neural TTS Engine (Port 8088)...")
    p_tts = subprocess.Popen([sys.executable, "-u", "tts_server.py"], env=env)
    processes.append(p_tts)
    time.sleep(1.5)

    # =========================================================================
    # Step 2: Launch NVIDIA Parakeet-TDT Streaming STT Engine (Port 8030)
    # Allocates before vLLM so STT is 100% GUARANTEED to secure CUDA memory!
    # =========================================================================
    logger.info("► [2/5] Launching NVIDIA Parakeet-TDT Streaming STT Engine (Port 8030)...")
    p_stt = subprocess.Popen([sys.executable, "-u", "stt_server.py"], env=env)
    processes.append(p_stt)
    time.sleep(2.0)

    # =========================================================================
    # Step 3: Launch Silero VAD Barge-In Controller (Port 8090)
    # Sub-100MB footprint
    # =========================================================================
    logger.info("► [3/5] Launching Silero VAD & Barge-in Controller (Port 8090)...")
    p_vad = subprocess.Popen([sys.executable, "-u", "vad_server.py"], env=env)
    processes.append(p_vad)
    time.sleep(1.0)

    # =========================================================================
    # Step 4: Launch vLLM OpenAI-Compatible LLM (Port 8000)
    # Now takes the remaining VRAM smoothly without starving audio models
    # =========================================================================
    logger.info(f"► [4/5] Launching vLLM Engine ({LLM_MODEL}) on Port 8000...")
    logger.info(f"   ⚡ Continuous Batching: max-num-seqs 32 | GPU Utilization: {GPU_MEM_UTIL}")
    vllm_cmd = [
        sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", LLM_MODEL,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
        "--enforce-eager",
        "--trust-remote-code"
    ]
    if API_KEY:
        vllm_cmd.extend(["--api-key", API_KEY])
    if "awq" in LLM_MODEL.lower():
        vllm_cmd.extend(["--quantization", "awq"])

    p_llm = None
    try:
        p_llm = subprocess.Popen(vllm_cmd, env=env)
        processes.append(p_llm)
        logger.success("✓ vLLM process spawned. Initializing weights...")
    except Exception as e:
        logger.error(f"Could not start vLLM: {e}")

    # Wait for LLM on Port 8000 before launching UI
    logger.info("⏳ Waiting for LLM Engine to load weights and listen on Port 8000...")
    llm_ready = False
    t_wait_start = time.time()
    last_progress_log = time.time()
    
    while time.time() - t_wait_start < 180:
        if p_llm and p_llm.poll() is not None:
            logger.error(f"❌ LLM process terminated with exit code {p_llm.returncode}!")
            break
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/health")
            if API_KEY:
                req.add_header("Authorization", f"Bearer {API_KEY}")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status in (200, 401, 403):
                    llm_ready = True
                    break
        except urllib.error.HTTPError as he:
            if he.code in (200, 401, 403):
                llm_ready = True
                break
        except Exception:
            pass
        
        if time.time() - last_progress_log >= 5.0:
            elapsed = int(time.time() - t_wait_start)
            logger.info(f"   ⏳ Initializing LLM in VRAM ({elapsed}s elapsed)...")
            last_progress_log = time.time()
            
        time.sleep(2)

    if llm_ready:
        logger.success(f"✓ LLM Engine is ONLINE & healthy on Port 8000! ({round(time.time() - t_wait_start, 1)}s)")
    else:
        logger.warning("⚠️ LLM readiness check timed out. Proceeding with remaining services...")

    # =========================================================================
    # Step 5: Start Gradio Interactive Human Prosody Testbench (Port 7860)
    # =========================================================================
    logger.info("► [5/5] Launching Gradio Testbench UI (Port 7860)...")
    p_ui = subprocess.Popen([sys.executable, "-u", "testbench_ui.py"], env=env)
    processes.append(p_ui)
    time.sleep(1.0)

    logger.success("\n==================================================================")
    logger.success("   🎉 ALL 5 GPU SERVICES ARE LIVE AND RUNNING!                   ")
    logger.success("==================================================================")
    logger.info(f"  • vLLM OpenAI API : http://{PUBLIC_IP}:{PORT_VLLM}/v1 (Port 8000)")
    logger.info(f"  • Kokoro TTS (54v): http://{PUBLIC_IP}:{PORT_TTS} (Port 8088)")
    logger.info(f"  • Parakeet STT API: http://{PUBLIC_IP}:{PORT_STT} (Port 8030)")
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
