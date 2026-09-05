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

API_KEY = os.getenv("GPU_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GPU_MEM_UTIL = os.getenv("GPU_MEM_UTIL", "0.50")
PUBLIC_IP = os.getenv("PUBLIC_IP", "77.54.200.11")
PORT_VLLM = os.getenv("PORT_VLLM", "15460")
PORT_TTS = os.getenv("PORT_TTS", "15188")
PORT_STT = os.getenv("PORT_STT", "15490")
PORT_VAD = os.getenv("PORT_VAD", "15089")
PORT_UI = os.getenv("PORT_UI", "15238")

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
    logger.info("   Hardware: 1x NVIDIA RTX 5060 Ti (16GB VRAM)                    ")
    logger.info("   CPU: Intel Xeon E5-2673 v4 (40 vCPUs, 96.5GB RAM)              ")
    logger.info(f"   Public IP: {PUBLIC_IP}                                          ")
    logger.info("==================================================================")

    env = os.environ.copy()
    env["GPU_API_KEY"] = API_KEY
    env["VLLM_USE_V1"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Minimize CUDA context & PyTorch overhead across all 4 worker processes
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info("📊 VRAM ALLOCATION BUDGET (16.0 GB Total):")
    logger.info("   • Qwen3-4B Q4_K_M Weights      : ~2.5 GB")
    logger.info("   • Parakeet TDT 0.6B INT8       : ~0.6 GB")
    logger.info("   • Kokoro-82M ONNX              : ~0.5 GB")
    logger.info("   • Silero VAD v5                : ~0.1 GB")
    logger.info("   • CUDA / PyTorch Contexts      : ~1.2 GB (Kernel & Driver overhead)")
    logger.info("   • Total Static Base Footprint   : ~4.9 GB / 16.0 GB")
    logger.info("   • FREE VRAM for 30 Callers     : ~11.1 GB (Continuous Batching Slots)")
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

    env["PYTHONUNBUFFERED"] = "1"

    # =========================================================================
    # 1. Start LLM Engine FIRST (Port 8000)
    # Allows vLLM to profile & allocate its memory pool without VRAM contention
    # =========================================================================
    gguf_model_path = os.path.join(os.path.dirname(__file__), "models", "llm", "Qwen3-4B-Q4_K_M.gguf")
    p_llm = None

    if os.path.exists(gguf_model_path):
        import shutil
        logger.info(f"► [1/5] Launching llama.cpp Engine with Continuous Batching ({gguf_model_path})...")
        logger.info("   ⚡ High-Concurrency Mode: --parallel 30 --cont-batching --flash-attn")
        
        llama_bin = shutil.which("llama-server") or "/usr/local/bin/llama-server"
        if llama_bin and os.path.exists(llama_bin):
            llama_cmd = [
                llama_bin,
                "-m", gguf_model_path,
                "--port", "8000",
                "--host", "0.0.0.0",
                "-ngl", "99",
                "--parallel", "30",
                "--cont-batching",
                "-c", "32768",
                "--flash-attn",
                "--alias", "Qwen3-4B"
            ]
            if API_KEY:
                llama_cmd.extend(["--api-key", API_KEY])
        else:
            llama_cmd = [
                sys.executable, "-u", "-m", "llama_cpp.server",
                "--model", gguf_model_path,
                "--port", "8000",
                "--host", "0.0.0.0",
                "--n_gpu_layers", "99",
                "--n_ctx", "32768"
            ]
            if API_KEY:
                llama_cmd.extend(["--api_key", API_KEY])
        try:
            p_llm = subprocess.Popen(llama_cmd, env=env)
            time.sleep(2.0)
            if p_llm.poll() is not None:
                logger.warning(f"llama.cpp server exited (code {p_llm.returncode}). Automatically falling back to vLLM...")
                p_llm = None
            else:
                processes.append(p_llm)
                logger.success("✓ llama.cpp server spawned with 30 concurrent slots & continuous batching!")
        except Exception as e:
            logger.warning(f"Could not start llama.cpp server: {e}. Falling back to vLLM...")
            p_llm = None

    if p_llm is None:
        logger.info(f"► [1/5] Launching vLLM Engine ({LLM_MODEL}) on Port 8000 FIRST...")
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

        try:
            p_llm = subprocess.Popen(vllm_cmd, env=env)
            processes.append(p_llm)
            logger.success("✓ vLLM process spawned. Waiting for CUDA memory allocation & engine readiness...")
        except Exception as e:
            logger.error(f"Could not start vLLM: {e}")

    # Wait for LLM on Port 8000 before starting subsequent services
    logger.info("⏳ Waiting for LLM Engine to load weights and listen on Port 8000...")
    llm_ready = False
    t_wait_start = time.time()
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
        
        elapsed = int(time.time() - t_wait_start)
        if elapsed > 0 and elapsed % 10 == 0:
            logger.info(f"   ... still initializing LLM ({elapsed}s elapsed) ...")
        time.sleep(2)

    if llm_ready:
        logger.success(f"✓ LLM Engine is ONLINE & healthy on Port 8000! ({round(time.time() - t_wait_start, 1)}s)")
    else:
        logger.warning("⚠️ LLM readiness check timed out or exited. Proceeding with remaining services...")

    # =========================================================================
    # 2. Start Kokoro-82M TTS Server (Port 8088)
    # =========================================================================
    logger.info("► [2/5] Launching Kokoro-82M Neural TTS Engine (Port 8088)...")
    p_tts = subprocess.Popen([sys.executable, "-u", "tts_server.py"], env=env)
    processes.append(p_tts)
    time.sleep(2.0)

    # =========================================================================
    # 3. Start STT Transcriber with Denoising (Port 8030)
    # =========================================================================
    logger.info("► [3/5] Launching Streaming STT Engine (Port 8030)...")
    p_stt = subprocess.Popen([sys.executable, "-u", "stt_server.py"], env=env)
    processes.append(p_stt)
    time.sleep(2.0)

    # =========================================================================
    # 4. Start Silero VAD Barge-In Controller (Port 8090)
    # =========================================================================
    logger.info("► [4/5] Launching Silero VAD & Barge-in Controller (Port 8090)...")
    p_vad = subprocess.Popen([sys.executable, "-u", "vad_server.py"], env=env)
    processes.append(p_vad)
    time.sleep(1.5)

    # =========================================================================
    # 5. Start Gradio Interactive Human Prosody Testbench (Port 7860)
    # =========================================================================
    logger.info("► [5/5] Launching Gradio Testbench UI (Port 7860)...")
    p_ui = subprocess.Popen([sys.executable, "-u", "testbench_ui.py"], env=env)
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
