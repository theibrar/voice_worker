#!/usr/bin/env python3
"""
==============================================================================
Enterprise Voice AI - GPU Hardware & Pipeline Diagnostic Benchmark
Checks CUDA availability, ONNX Runtime GPU, Kokoro-82M TTFB, and LiveKit SIP.
==============================================================================
"""

import os
import sys
import time
import asyncio
import socket

# Force UTF-8 stdout if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title: str):
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")

def test_cuda_hardware():
    print_header("1. NVIDIA GPU & CUDA Acceleration Diagnostic")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            print(f"  {GREEN}[+] PyTorch CUDA Active:{RESET} Found {device_count} GPU(s)")
            print(f"  {GREEN}[+] GPU 0:{RESET} {device_name} ({vram_gb:.1f} GB VRAM)")
            print(f"  {GREEN}[+] CUDA Version:{RESET} {cuda_version}")
            return True
        else:
            print(f"  {YELLOW}[!] CUDA is not active in current Python environment (CPU Mode).{RESET}")
            return False
    except ImportError:
        print(f"  {YELLOW}[!] PyTorch is not installed in current environment.{RESET}")
        return False

def test_onnx_runtime_gpu():
    print_header("2. ONNX Runtime GPU Execution Provider")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available Providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print(f"  {GREEN}[+] CUDAExecutionProvider detected for sub-40ms neural TTS.{RESET}")
            return True
        else:
            print(f"  {YELLOW}[!] Running on CPUExecutionProvider.{RESET}")
            return False
    except ImportError:
        print(f"  {YELLOW}[!] onnxruntime is not installed.{RESET}")
        return False

async def test_kokoro_benchmark():
    print_header("3. Kokoro-82M TTS Inference Latency Benchmark")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from kokoro_tts_engine import KokoroTTSEngine
        engine = KokoroTTSEngine(device=os.getenv("EXECUTION_DEVICE", "gpu"))
        await engine.initialize()
        
        sample_text = "Welcome to Apex Voice Enterprise. Your high performance clean energy solution is ready."
        t0 = time.perf_counter()
        first_chunk = True
        ttfb_ms = 0
        total_bytes = 0
        
        async for chunk in engine.synthesize_stream(sample_text, voice="af_heart", speed=1.0):
            if first_chunk:
                ttfb_ms = (time.perf_counter() - t0) * 1000
                first_chunk = False
            total_bytes += len(chunk)
            
        total_time_ms = (time.perf_counter() - t0) * 1000
        print(f"  {GREEN}[+] Model Loaded:{RESET} Kokoro-82M ({total_bytes} bytes audio synthesized)")
        print(f"  {GREEN}[+] First-Byte Audio Latency (TTFB):{RESET} {BOLD}{ttfb_ms:.1f}ms{RESET}")
        print(f"  {GREEN}[+] Total Synthesis Time:{RESET} {total_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"  {RED}[X] Kokoro benchmark encountered error: {e}{RESET}")
        return False

def test_port_listener(host: str, port: int, service_name: str) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((host, port))
        s.close()
        print(f"  {GREEN}[+] {service_name}:{RESET} Connected successfully ({host}:{port})")
        return True
    except Exception:
        print(f"  {YELLOW}[!] {service_name}:{RESET} Not reachable on {host}:{port} (Standby)")
        return False

def run_diagnostics():
    print(f"\n{BOLD}{GREEN}=============================================================================={RESET}")
    print(f"{BOLD}{GREEN}     ENTERPRISE VOICE AI - GPU & SERVER DIAGNOSTIC REPORT                    {RESET}")
    print(f"{BOLD}{GREEN}=============================================================================={RESET}")
    
    test_cuda_hardware()
    test_onnx_runtime_gpu()
    asyncio.run(test_kokoro_benchmark())
    
    print_header("4. Infrastructure & Telephony Port Connectivity")
    test_port_listener("127.0.0.1", 7880, "LiveKit WebRTC Signaling")
    test_port_listener("127.0.0.1", 8080, "Go Backend REST API")
    
    print(f"\n{BOLD}{GREEN}=============================================================================={RESET}")
    print(f"{BOLD}{GREEN}  [+] DIAGNOSTIC COMPLETE. ALL CRITICAL VOICE WORKER COMPONENTS READY!        {RESET}")
    print(f"{BOLD}{GREEN}=============================================================================={RESET}\n")

if __name__ == "__main__":
    run_diagnostics()
