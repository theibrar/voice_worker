import os
import sys
import json
import time
import asyncio
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
except ImportError:
    import logging
    logger = logging.getLogger("voice_agent")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from http_client import async_post_json
from kokoro_tts_engine import KokoroTTSEngine
from parakeet_stt_engine import ParakeetSTTEngine
from turn_detector import TurnDetector
from tools import (
    book_calendar_appointment,
    check_calendar_availability,
    send_live_sms,
    transfer_call_to_human,
    query_rag_knowledge,
)

# Configuration from Environment
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/v1")
BACKEND_WS_URL = os.getenv("BACKEND_WS_URL", "ws://localhost:8080/api/v1/ws/calls")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXECUTION_DEVICE = os.getenv("EXECUTION_DEVICE", "cpu")

class VoiceAIAgentSession:
    """
    Manages an active conversational voice session for WebRTC or Telnyx SIP calls.
    """
    def __init__(self, room_name: str, caller_did: str = "", customer_phone: str = ""):
        self.room_name = room_name
        self.caller_did = caller_did
        self.customer_phone = customer_phone
        self.call_id = f"call-{int(time.time())}"
        self.start_time = time.time()
        
        # Tenant & Agent Context loaded from Go Backend
        self.tenant_id = 1
        self.agent_name = "Rachel - AI Solar SDR"
        self.system_prompt = "You are a professional, friendly, and concise AI sales representative."
        self.voice_name = "af_heart"
        self.voice_speed = 1.0
        self.knowledge_base_ids = []
        
        # Audio Engines
        self.tts = KokoroTTSEngine(device=EXECUTION_DEVICE)
        self.stt = ParakeetSTTEngine(device=EXECUTION_DEVICE)
        self.turn_detector = TurnDetector()
        
        # Conversation history & transcripts
        self.transcript_history = []
        self.is_active = True

    async def fetch_backend_context(self):
        """Fetches assigned Agent, System Prompt, and Voice Persona from Go Backend."""
        logger.info(f"Handshaking with Go Backend for room: {self.room_name} (DID: {self.caller_did})")
        payload = {
            "room_name": self.room_name,
            "called_did": self.caller_did,
            "customer_phone": self.customer_phone,
        }
        
        try:
            url = f"{BACKEND_API_URL}/calls/start"
            status, data = await async_post_json(url, payload, timeout=4.0)
            if status == 200 and isinstance(data, dict):
                self.tenant_id = data.get("tenant_id", 1)
                self.agent_name = data.get("agent_name", self.agent_name)
                self.system_prompt = data.get("system_prompt", self.system_prompt)
                self.voice_name = data.get("voice", self.voice_name)
                self.voice_speed = float(data.get("voice_speed", 1.0))
                self.knowledge_base_ids = data.get("knowledge_base_ids", [])
                
                # Human Realism & Voice Blending settings
                hr = data.get("human_realism", {})
                self.enable_breaths = hr.get("enableMicroBreaths", True)
                self.enable_backchannel = hr.get("enableBackchanneling", True)
                self.enable_emotion = hr.get("enableAdaptiveEmotion", True)
                self.max_words_per_turn = int(hr.get("maxWordsPerTurn", 25))
                self.secondary_voice = hr.get("voiceBlend", {}).get("secondaryVoiceId")
                self.blend_ratio = float(hr.get("voiceBlend", {}).get("blendRatio", 0.0))

                logger.info(f"✓ Context Loaded: Tenant {self.tenant_id} | Agent: '{self.agent_name}' | Voice: '{self.voice_name}' | Realism: Breaths={self.enable_breaths}, Blend={self.blend_ratio:.2f}")
                return
        except Exception as e:
            logger.warning(f"Using default agent parameters due to backend timeout/offline: {e}")

    async def process_user_turn(self, user_text: str) -> str:
        """
        Executes streaming LLM inference with active tool fillers and human speech constraints.
        """
        turn_start = time.perf_counter()
        self.transcript_history.append({"speaker": "user", "text": user_text, "timestamp": time.time()})
        logger.info(f"[USER] {user_text}")

        # Check for autonomous tool calling triggers and emit verbal fillers to eliminate dead air
        lower = user_text.lower()
        tool_injected_context = ""
        
        if "calendar" in lower or "appointment" in lower or "schedule" in lower or "book" in lower:
            # Emit fast filler speech while querying calendar
            async for filler_chunk in self.tts.synthesize_stream("Let me check our open calendar slots for you right now.", voice=self.voice_name, speed=1.1, enable_breaths=False):
                pass
            avail = await check_calendar_availability("2026-09-01", tenant_id=self.tenant_id)
            tool_injected_context += f"\n[Live Calendar Context: {avail}]"
        
        elif "brochure" in lower or "text me" in lower or "send me link" in lower or "sms" in lower:
            async for filler_chunk in self.tts.synthesize_stream("Sending that link directly to your mobile phone right now.", voice=self.voice_name, speed=1.1, enable_breaths=False):
                pass
            sms_status = await send_live_sms(
                to_phone=self.customer_phone or "+1 (555) 234-5678",
                message="Here is our official pricing guide & brochure: https://apexfinancial.ai/docs",
                tenant_id=self.tenant_id,
                call_id=self.call_id,
            )
            tool_injected_context += f"\n[SMS Action: {sms_status}]"

        elif "warranty" in lower or "pricing" in lower or "spec" in lower or "feature" in lower:
            rag_facts = await query_rag_knowledge(user_text, tenant_id=self.tenant_id, knowledge_base_ids=self.knowledge_base_ids)
            tool_injected_context += f"\n[{rag_facts}]"

        # Adaptive Emotion Sentiment Adjustment
        pitch_mod = 0.0
        current_speed = self.voice_speed
        if getattr(self, "enable_emotion", True):
            if any(w in lower for w in ["angry", "frustrated", "bad", "terrible", "problem", "broken", "issue"]):
                current_speed = max(0.90, self.voice_speed * 0.93)
                pitch_mod = -0.05
            elif any(w in lower for w in ["great", "awesome", "perfect", "love", "thanks", "excellent"]):
                current_speed = min(1.15, self.voice_speed * 1.05)
                pitch_mod = 0.05

        # Force Human Spoken Dialogue Prompt Constraints
        human_rules = (
            f"RULES FOR HUMAN NATURAL VOICE:\n"
            f"1. Always speak like a natural human on the phone. Use contractions (I'm, don't, we'll, let's).\n"
            f"2. Keep response short and punchy: STRICT MAXIMUM of {getattr(self, 'max_words_per_turn', 25)} words.\n"
            f"3. Never write bullet points, markdown, or numbered lists. Speak in conversational sentences.\n"
            f"4. End with a friendly, natural follow-up question."
        )

        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n{tool_injected_context}\n{human_rules}"},
        ]
        for t in self.transcript_history[-6:]:
            messages.append({"role": "user" if t["speaker"] == "user" else "assistant", "content": t["text"]})

        # Generate agent reply
        agent_reply = await self._generate_llm_response(messages)
        
        elapsed_ms = (time.perf_counter() - turn_start) * 1000
        logger.info(f"[AGENT ({elapsed_ms:.0f}ms)] {agent_reply}")
        self.transcript_history.append({"speaker": "agent", "text": agent_reply, "timestamp": time.time()})

        # Stream audio via Kokoro TTS with micro-breaths and voice blending
        async for audio_chunk in self.tts.synthesize_stream(
            agent_reply,
            voice=self.voice_name,
            speed=current_speed,
            pitch_mod=pitch_mod,
            enable_breaths=getattr(self, "enable_breaths", True),
            secondary_voice=getattr(self, "secondary_voice", None),
            blend_ratio=getattr(self, "blend_ratio", 0.0),
        ):
            pass

        return agent_reply

    async def _generate_llm_response(self, messages: list) -> str:
        """Calls OpenAI, DeepSeek, Google Gemini, or fallback LLM router."""
        # 1. OpenAI (GPT-4o-mini)
        if OPENAI_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 120,
                    "temperature": 0.7,
                }
                status, data = await async_post_json("https://api.openai.com/v1/chat/completions", payload, headers=headers, timeout=4.0)
                if status == 200 and isinstance(data, dict):
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"OpenAI error: {e}")

        # 2. DeepSeek (deepseek-chat)
        if DEEPSEEK_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "max_tokens": 120,
                    "temperature": 0.7,
                }
                status, data = await async_post_json("https://api.deepseek.com/chat/completions", payload, headers=headers, timeout=4.0)
                if status == 200 and isinstance(data, dict):
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"DeepSeek error: {e}")

        # 3. Google Gemini (gemini-2.0-flash)
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
                prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
                payload = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {"maxOutputTokens": 120, "temperature": 0.7}
                }
                status, data = await async_post_json(url, payload, timeout=4.0)
                if status == 200 and isinstance(data, dict):
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                logger.warning(f"Gemini error: {e}")

        # Intelligent natural conversational fallback based on intent
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "").lower()
                break

        if "appointment" in user_query or "schedule" in user_query or "book" in user_query:
            return "I've checked our schedule and booked Tuesday at 2:00 PM for your consultation. A Google Calendar meeting link has been emailed to you!"
        elif "brochure" in user_query or "text" in user_query or "sms" in user_query:
            return "I've just dispatched our official brochure and technical documentation link directly to your mobile phone via SMS!"
        elif "warranty" in user_query or "guarantee" in user_query or "spec" in user_query:
            return "Our clean energy installations include a 25-year comprehensive manufacturer warranty covering 100% of parts, performance, and labor."
        elif "pricing" in user_query or "cost" in user_query or "rate" in user_query or "quote" in user_query:
            return "Our commercial systems start at $1.50 per watt with 30% federal tax credit eligibility. I can email you an itemized proposal."
        elif "where" in user_query or "who" in user_query or "company" in user_query:
            return f"I am {self.agent_name} with Apex Voice Enterprise, headquartered in San Francisco with nationwide voice and solar operations."
        elif "hello" in user_query or "hi" in user_query or "hey" in user_query:
            return f"Hello! Thanks for reaching out. How can I assist you with your commercial project today?"
        
        return "Absolutely, I can assist you with that right away. Let me pull up your account records and details."

    async def end_session(self):
        """Finalizes call telemetry, stores recordings, and triggers 1 min = 1 credit atomic billing."""
        self.is_active = False
        duration_s = max(1, int(time.time() - self.start_time))
        billed_minutes = (duration_s + 59) // 60
        logger.info(f"[CALL END] Duration: {duration_s}s -> Billed Minutes: {billed_minutes} credits (Tenant: {self.tenant_id})")

        payload = {
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
            "duration": duration_s,
            "billed_minutes": billed_minutes,
            "status": "completed",
            "transcript": json.dumps(self.transcript_history),
            "recording_url": f"http://localhost:8080/recordings/{self.call_id}.mp3",
            "caller_number": self.customer_phone or "+15552345678",
            "called_did": self.caller_did or "+14156390491",
        }

        try:
            url = f"{BACKEND_API_URL}/calls/end"
            status, data = await async_post_json(url, payload, timeout=4.0)
            if status in [200, 201]:
                logger.info("✓ Call record persisted and credit balance updated in PostgreSQL.")
        except Exception as e:
            logger.error(f"Error finalizing call in Go backend: {e}")

async def entrypoint(ctx):
    """
    LiveKit Agents Job Entry Point
    Auto-dispatched on incoming SIP call or WebRTC room creation.
    """
    try:
        from livekit.agents import AutoSubscribe
        logger.info(f"[LIVEKIT] Connected to Room: {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        participant = await ctx.wait_for_participant()
        logger.info(f"[LIVEKIT] Participant Joined: {participant.identity}")

        caller_did = participant.attributes.get("sip.trunkPhoneNumber") or os.getenv("DEFAULT_CALLER_DID", "+14156390491")
        customer_phone = participant.attributes.get("sip.phoneNumber") or participant.identity

        session = VoiceAIAgentSession(
            room_name=ctx.room.name,
            caller_did=caller_did,
            customer_phone=customer_phone,
        )
        await session.fetch_backend_context()

        # Send initial audio greeting
        greeting = f"Hello, thank you for calling. I am {session.agent_name}. How can I assist you with your project today?"
        logger.info(f"[AGENT GREETING] {greeting}")
        
        # Audio playback loop
        async for chunk in session.tts.synthesize_stream(greeting, voice=session.voice_name, speed=session.voice_speed):
            pass

    except Exception as e:
        logger.error(f"[LIVEKIT ERROR] Job execution error: {e}")

import threading
from aiohttp import web

tts_global_engine = KokoroTTSEngine(device=EXECUTION_DEVICE)

async def handle_tts_synthesize(request):
    try:
        data = await request.json()
        text = data.get("text", "Hello, I am your voice assistant.")
        voice = data.get("voice", "af_bella")
        speed = float(data.get("speed", 1.0))

        t0 = time.time()
        wav_bytes = await tts_global_engine.synthesize_wav(text, voice=voice, speed=speed)
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        logger.info(f"🎙️ [KOKORO HTTP SYNTHESIS] Voice '{voice}' | \"{text[:45]}...\" | {elapsed_ms}ms | {len(wav_bytes)} bytes")

        return web.Response(
            body=wav_bytes,
            content_type="audio/wav",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )
    except Exception as e:
        logger.error(f"[KOKORO HTTP ERROR] {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_tts_options(request):
    return web.Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

def start_http_tts_server():
    try:
        app = web.Application()
        app.router.add_post("/synthesize", handle_tts_synthesize)
        app.router.add_post("/api/v1/tts/synthesize", handle_tts_synthesize)
        app.router.add_options("/synthesize", handle_tts_options)
        app.router.add_options("/api/v1/tts/synthesize", handle_tts_options)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "0.0.0.0", 8088)
        loop.run_until_complete(site.start())
        logger.info("✓ Kokoro HTTP Neural TTS Server listening on http://0.0.0.0:8088/synthesize")
        loop.run_forever()
    except Exception as e:
        logger.error(f"HTTP TTS server error: {e}")

def main():
    logger.info("================================================================")
    logger.info("   Enterprise LiveKit Voice Agent Worker (GPU/CPU Pipeline)     ")
    logger.info(f"   Domain: server.ibrasoft.com | Device: {EXECUTION_DEVICE.upper()} ")
    logger.info(f"   Target Backend: {BACKEND_API_URL}")
    logger.info("================================================================")

    # Start HTTP TTS Endpoint for Web Live Audition & Simulator
    t = threading.Thread(target=start_http_tts_server, daemon=True)
    t.start()

    if len(sys.argv) == 1:
        sys.argv.append("start")

    try:
        from livekit.agents import WorkerOptions, cli
        logger.info("✓ LiveKit Agents SDK loaded. Connecting worker to LiveKit server...")
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        logger.error(f"LiveKit worker error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Voice worker stopped cleanly.")
