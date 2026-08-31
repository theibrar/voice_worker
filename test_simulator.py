import os
import sys
import time
import asyncio

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("test_simulator")

from kokoro_tts_engine import KokoroTTSEngine
from parakeet_stt_engine import ParakeetSTTEngine

EXECUTION_DEVICE = os.getenv("EXECUTION_DEVICE", "cpu")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

async def run_simulation():
    print("\n" + "=" * 65)
    print(" 🎙️  ENTERPRISE GPU VOICE AGENT - LIVE INTERACTIVE SIMULATOR")
    print(f"    Device: {EXECUTION_DEVICE.upper()} | Kokoro-82M TTS + Parakeet STT")
    print("=" * 65 + "\n")

    print("► Initializing Kokoro-82M Neural Audio Engine...")
    tts = KokoroTTSEngine(device=EXECUTION_DEVICE)
    await tts.initialize()
    print("✓ Kokoro-82M loaded successfully!\n")

    agent_name = "Rachel - AI Solar SDR"
    voice_name = "af_bella"
    system_prompt = "You are a professional, friendly, and concise AI sales representative."

    # Test initial greeting synthesis
    greeting = f"Hello! Thanks for calling Apex Voice. My name is {agent_name}. How can I assist you with your project today?"
    print(f"[{agent_name}]: {greeting}")
    
    t0 = time.perf_counter()
    wav_bytes = await tts.synthesize_wav(greeting, voice=voice_name, speed=1.0)
    audio_ms = (time.perf_counter() - t0) * 1000
    print(f"└── [KOKORO TTS (GPU/CPU)]: Synthesized {len(wav_bytes)} bytes in {audio_ms:.1f}ms\n")

    print("-" * 65)
    print("► CONVERSATION ACTIVE! Type your message to test the agent (or 'exit' to quit):")
    print("-" * 65)

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, "\n[You]: ")
            user_input = user_input.strip()

            if not user_input or user_input.lower() in ["exit", "quit", "bye", "hangup"]:
                print("\n[Simulator] Ending conversation session.")
                break

            # 1. LLM Reasoning
            t_llm = time.perf_counter()
            reply = ""

            # Call DeepSeek if available
            if DEEPSEEK_API_KEY:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.deepseek.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                            json={
                                "model": "deepseek-chat",
                                "messages": [
                                    {"role": "system", "content": f"{system_prompt}\nKeep responses conversational and under 25 words."},
                                    {"role": "user", "content": user_input}
                                ],
                                "max_tokens": 100,
                            },
                            timeout=aiohttp.ClientTimeout(total=4.0)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                except Exception:
                    pass

            if not reply:
                # Intelligent local fallback
                lower = user_input.lower()
                if "name" in lower or "who are you" in lower:
                    reply = f"I'm {agent_name}, powered by Kokoro-82M neural voice on server.ibrasoft.com."
                elif "price" in lower or "cost" in lower:
                    reply = "Our pricing starts at $0.08 per minute with full Kokoro acceleration and zero per-seat fees."
                elif "demo" in lower or "schedule" in lower or "meeting" in lower:
                    reply = "I have tomorrow at 2:00 PM available. Would you like me to book that for you?"
                else:
                    reply = f"I understand! Regarding '{user_input}', I can assist with that right away. What details do you need?"

            llm_ms = (time.perf_counter() - t_llm) * 1000

            # 2. Kokoro Neural TTS Synthesis
            t_tts = time.perf_counter()
            wav = await tts.synthesize_wav(reply, voice=voice_name, speed=1.0)
            tts_ms = (time.perf_counter() - t_tts) * 1000

            total_ms = llm_ms + tts_ms

            print(f"\n[{agent_name}]: {reply}")
            print(f"└── ⚡ [BENCHMARK]: LLM Reasoning: {llm_ms:.1f}ms | Kokoro Neural TTS: {tts_ms:.1f}ms | Total Roundtrip: {total_ms:.1f}ms")

        except (KeyboardInterrupt, EOFError):
            break

    print("\n✓ Simulator Completed Successfully!\n")

if __name__ == "__main__":
    asyncio.run(run_simulation())
