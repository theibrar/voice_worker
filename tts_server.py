"""
Kokoro-82M Streaming Neural TTS Server (Port 8088)
- Ultra-Low Latency Chunked Streaming (<150ms TTFA)
- OpenAI-Compatible /v1/audio/speech Endpoint
- Emotion & Prosody Control Tags ([empathy], [cheerful], [urgent], [calm])
- Cognitive Fillers & Thinking Foley Bridge
"""

import os
import sys
import ctypes

# Preload NVIDIA CUDA / cuBLAS libraries into process before ONNX init
nvidia_dirs = [
    "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib"
]
for d in nvidia_dirs:
    if os.path.exists(d):
        if d not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = f"{d}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        for lib_name in ["libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"]:
            lib_path = os.path.join(d, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

import io
import re
import time
import asyncio
import numpy as np
import soundfile as sf
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

API_KEY = os.getenv("GPU_API_KEY", "sk-ibrasoft-gpu-voice")
MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "/app/models/kokoro-v0_19.onnx")
VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "/app/models/voices.bin")

# Fallback paths for local directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "kokoro-v0_19.onnx")
if not os.path.exists(VOICES_PATH):
    VOICES_PATH = os.path.join(os.path.dirname(__file__), "models", "voices.bin")

app = FastAPI(title="Kokoro Neural Streaming TTS Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kokoro_engine = None

def get_kokoro():
    global kokoro_engine
    if kokoro_engine is None:
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            providers = getattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])()
            use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
            logger.info(f"Loading Kokoro-82M ONNX with providers: {use_providers}...")
            
            if hasattr(ort, "InferenceSession"):
                inf_sess = ort.InferenceSession(MODEL_PATH, providers=use_providers)
                kokoro_engine = Kokoro.from_session(inf_sess, VOICES_PATH)
            else:
                kokoro_engine = Kokoro(MODEL_PATH, VOICES_PATH)
            logger.success("✓ Kokoro-82M Neural TTS initialized successfully.")
        except Exception as e:
            logger.warning(f"ONNX session init notice: {e}. Falling back to standard Kokoro loader...")
            try:
                from kokoro_onnx import Kokoro
                kokoro_engine = Kokoro(MODEL_PATH, VOICES_PATH)
                logger.success("✓ Kokoro-82M Neural TTS initialized.")
            except Exception as e2:
                logger.error(f"Failed to load Kokoro ONNX: {e2}")
    return kokoro_engine

# Authentication Helper
def verify_api_key(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if API_KEY and API_KEY != "":
        if not auth_header.startswith("Bearer "):
            # Allow query param fallback
            api_param = request.query_params.get("api_key", "")
            if api_param != API_KEY:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
        else:
            token = auth_header.replace("Bearer ", "").strip()
            if token != API_KEY:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# Request Schemas
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0
    lang: Optional[str] = "en-us"
    stream: Optional[bool] = False

class OpenAISpeechRequest(BaseModel):
    model: Optional[str] = "kokoro"
    input: str
    voice: Optional[str] = "af_bella"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0

# Emotion Tag Processor
def parse_emotion_and_prosody(text: str, base_speed: float = 1.0):
    """
    Extracts emotional prosody tags:
    [empathy] -> slightly slower (0.92x), softer delivery
    [cheerful] -> slightly faster (1.05x), energetic
    [urgent] -> faster (1.18x), direct
    [whisper] / [calm] -> slower (0.88x)
    """
    speed = base_speed
    emotion = "neutral"

    if "[empathy]" in text.lower():
        speed = max(0.85, base_speed * 0.92)
        emotion = "empathy"
    elif "[cheerful]" in text.lower():
        speed = min(1.3, base_speed * 1.06)
        emotion = "cheerful"
    elif "[urgent]" in text.lower():
        speed = min(1.4, base_speed * 1.18)
        emotion = "urgent"
    elif "[calm]" in text.lower() or "[whisper]" in text.lower():
        speed = max(0.82, base_speed * 0.88)
        emotion = "calm"

    # Clean tags from spoken text
    clean_text = re.sub(r"\[(empathy|cheerful|urgent|calm|whisper|neutral)\]", "", text, flags=re.IGNORECASE).strip()
    return clean_text, speed, emotion

@app.on_event("startup")
async def startup_event():
    get_kokoro()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kokoro-tts",
        "engine_ready": kokoro_engine is not None,
        "sample_rate": 24000,
        "default_voice": "af_bella",
    }

@app.post("/synthesize")
async def synthesize_speech(req: SynthesizeRequest, request: Request):
    verify_api_key(request)
    kokoro = get_kokoro()
    if not kokoro:
        raise HTTPException(status_code=500, detail="Kokoro TTS engine not initialized.")

    clean_text, effective_speed, emotion = parse_emotion_and_prosody(req.text, req.speed or 1.0)
    voice_name = req.voice or "af_bella"

    t0 = time.time()
    try:
        samples, sample_rate = kokoro.create(clean_text, voice=voice_name, speed=effective_speed, lang=req.lang or "en-us")
        
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        latency_ms = round((time.time() - t0) * 1000, 1)
        logger.info(f"🎙️ [KOKORO TTS] voice='{voice_name}' emotion='{emotion}' | {latency_ms}ms | {len(wav_bytes)} bytes")

        return Response(content=wav_bytes, media_type="audio/wav", headers={
            "X-Latency-Ms": str(latency_ms),
            "X-Emotion-Detected": emotion,
            "X-Sample-Rate": str(sample_rate),
        })
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI-Compatible /v1/audio/speech
@app.post("/v1/audio/speech")
async def openai_compatible_speech(req: OpenAISpeechRequest, request: Request):
    verify_api_key(request)
    kokoro = get_kokoro()
    if not kokoro:
        raise HTTPException(status_code=500, detail="Kokoro TTS not ready")

    clean_text, effective_speed, emotion = parse_emotion_and_prosody(req.input, req.speed or 1.0)
    
    # Map common OpenAI voice aliases to Kokoro
    voice_map = {
        "alloy": "af_bella",
        "echo": "am_michael",
        "fable": "af_sarah",
        "onyx": "am_adam",
        "nova": "af_heart",
        "shimmer": "bf_emma",
    }
    target_voice = voice_map.get(req.voice.lower(), req.voice)

    samples, sample_rate = kokoro.create(clean_text, voice=target_voice, speed=effective_speed, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    
    return Response(content=buf.getvalue(), media_type="audio/wav")

# Streaming Chunked Endpoint (TTFA < 150ms)
@app.post("/stream")
async def stream_speech(req: SynthesizeRequest, request: Request):
    verify_api_key(request)
    kokoro = get_kokoro()
    if not kokoro:
        raise HTTPException(status_code=500, detail="Kokoro TTS not ready")

    clean_text, effective_speed, _ = parse_emotion_and_prosody(req.text, req.speed or 1.0)
    voice_name = req.voice or "af_bella"

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        # Split into short conversational sentences for sub-150ms first-chunk delivery
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        if not sentences:
            sentences = [clean_text]

        for s in sentences:
            samples, sr = kokoro.create(s, voice=voice_name, speed=effective_speed, lang=req.lang or "en-us")
            # Convert float32 samples to 16-bit PCM
            pcm16 = (samples * 32767).astype(np.int16).tobytes()
            yield pcm16
            await asyncio.sleep(0.01)

    return StreamingResponse(audio_generator(), media_type="application/octet-stream", headers={
        "Transfer-Encoding": "chunked",
        "Content-Type": "audio/pcm; rate=24000; channels=1",
    })

# Cognitive Fillers & Thinking Foley
FILLER_PHRASES = [
    "Let me pull that up for you...",
    "One second, checking the records right now...",
    "Certainly, looking into that for you...",
    "Give me just a brief moment while I verify that...",
]

@app.get("/filler")
async def get_cognitive_filler(voice: Optional[str] = "af_bella", request: Request = None):
    kokoro = get_kokoro()
    phrase = np.random.choice(FILLER_PHRASES)
    samples, sr = kokoro.create(phrase, voice=voice, speed=1.05)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav", headers={
        "X-Filler-Phrase": phrase
    })

# WebSocket Real-Time Token-to-Audio Streaming
@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    kokoro = get_kokoro()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice = data.get("voice", "af_bella")
            speed = float(data.get("speed", 1.0))

            if text.strip() and kokoro:
                clean_text, eff_speed, _ = parse_emotion_and_prosody(text, speed)
                samples, sr = kokoro.create(clean_text, voice=voice, speed=eff_speed)
                pcm16 = (samples * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm16)
    except WebSocketDisconnect:
        logger.info("WebSocket TTS client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket TTS error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088, access_log=False)
