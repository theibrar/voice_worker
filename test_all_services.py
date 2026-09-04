import sys
import time
import json
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Master Configuration
HOST = "184.144.154.180"
API_KEY = "sk-ibrasoft-gpu-voice"

PORTS = {
    "LLM": 56137,
    "TTS": 56209,
    "STT": 56546,
    "VAD": 56756,
    "UI":  56081
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{CYAN}{BOLD}   >> APEX GPU VOICE AI CLUSTER - END-TO-END HEALTH & LATENCY TEST      {RESET}")
    print(f"{CYAN}   Target Host: {HOST} | Auth: Bearer {API_KEY[:8]}...{RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}\n")

results = []

def test_endpoint(name, url, method="GET", headers=None, data=None):
    if headers is None:
        headers = {}
    
    t0 = time.time()
    status_str = "FAIL"
    latency_ms = 0
    detail = ""

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=12) as response:
            latency_ms = round((time.time() - t0) * 1000, 1)
            code = response.getcode()
            body = response.read()
            if code == 200:
                status_str = "PASS"
                try:
                    parsed = json.loads(body.decode('utf-8'))
                    if "choices" in parsed:
                        detail = f"Generated: \"{parsed['choices'][0]['message']['content'].strip()}\""
                    elif "features" in parsed or "status" in parsed:
                        detail = f"Status: {parsed.get('status', 'OK')} | Service: {parsed.get('service', 'active')}"
                    elif "is_speech" in parsed:
                        detail = f"Speech Prob: {parsed.get('probability', 0.0)}"
                    else:
                        detail = f"Response size: {len(body)} bytes"
                except Exception:
                    detail = f"Binary/HTML: {len(body)} bytes received"
            else:
                detail = f"HTTP {code}"
    except urllib.error.HTTPError as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        detail = f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        detail = str(e)

    results.append({
        "name": name,
        "url": url,
        "status": status_str,
        "latency_ms": latency_ms,
        "detail": detail
    })
    return status_str == "PASS"

def main():
    print_banner()

    # 1. Test LLM Models & Chat Completion
    print(f"► Testing [1/5] vLLM OpenAI Engine (Port {PORTS['LLM']})...")
    llm_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    test_endpoint("LLM Models API", f"http://{HOST}:{PORTS['LLM']}/v1/models", headers=llm_headers)
    
    chat_payload = json.dumps({
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Say hello in 3 words."}],
        "max_tokens": 15
    }).encode('utf-8')
    test_endpoint("LLM Chat Stream", f"http://{HOST}:{PORTS['LLM']}/v1/chat/completions", method="POST", headers=llm_headers, data=chat_payload)

    # 2. Test Kokoro Neural TTS
    print(f"► Testing [2/5] Kokoro-82M Neural TTS (Port {PORTS['TTS']})...")
    tts_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    test_endpoint("Kokoro TTS Health", f"http://{HOST}:{PORTS['TTS']}/health")
    
    tts_payload = json.dumps({
        "text": "[cheerful] Hello! System is online.",
        "voice": "am_michael",
        "speed": 1.0
    }).encode('utf-8')
    test_endpoint("Kokoro Synthesis", f"http://{HOST}:{PORTS['TTS']}/synthesize", method="POST", headers=tts_headers, data=tts_payload)

    # 3. Test Faster-Whisper STT
    print(f"► Testing [3/5] Faster-Whisper CUDA STT (Port {PORTS['STT']})...")
    test_endpoint("STT Health Probe", f"http://{HOST}:{PORTS['STT']}/health")

    # 4. Test Silero VAD
    print(f"► Testing [4/5] Silero VAD Barge-In Engine (Port {PORTS['VAD']})...")
    test_endpoint("Silero VAD Health", f"http://{HOST}:{PORTS['VAD']}/health")

    # 5. Test Gradio Web Playground
    print(f"► Testing [5/5] Gradio UI Web Server (Port {PORTS['UI']})...")
    test_endpoint("Gradio UI Web", f"http://{HOST}:{PORTS['UI']}")

    # Output Summary Table
    print(f"\n{BOLD}========================================================================{RESET}")
    print(f"{'SERVICE':<22} | {'STATUS':<6} | {'LATENCY':<9} | {'DETAILS'}")
    print(f"-----------------------+--------+-----------+---------------------------{RESET}")
    
    all_pass = True
    for r in results:
        status_color = GREEN if r['status'] == "PASS" else RED
        latency_color = GREEN if r['latency_ms'] < 300 else (YELLOW if r['latency_ms'] < 1000 else RED)
        status_badge = f"{status_color}{BOLD}{r['status']:<6}{RESET}"
        latency_badge = f"{latency_color}{r['latency_ms']} ms{RESET}"
        print(f"{r['name']:<22} | {status_badge} | {latency_badge:<18} | {r['detail'][:40]}")
        if r['status'] != "PASS":
            all_pass = False

    print(f"{BOLD}========================================================================{RESET}")
    if all_pass:
        print(f"\n{GREEN}{BOLD}[SUCCESS] ALL GPU SERVICES ARE 100% OPERATIONAL & READY FOR CALLS!{RESET}\n")
    else:
        print(f"\n{YELLOW}{BOLD}[NOTICE] One or more services are still initializing or unreachable. Check details above.{RESET}\n")

if __name__ == "__main__":
    main()
