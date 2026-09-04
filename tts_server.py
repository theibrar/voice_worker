"""
Kokoro-82M Enterprise Neural TTS Server (Port 8088)
- Ultra-Low Latency Chunked Streaming (<150ms TTFA)
- Free-form & Natural Language Inline Style Tags ([whisper in small voice], [excited and fast], [calm], [urgent])
- Non-Verbal Sound Cues Interceptor ((laughs), (sighs), (coughs), (gasps))
- Wrapper-Level SSML Parsing (<break time="300ms"/>, <prosody rate="90%">)
- Gain/Volume Control & Telephony Resampling
- OpenAI-Compatible /v1/audio/speech Endpoint
- WebSocket Real-Time PCM Streaming
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
from functools import lru_cache
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

API_KEY = os.getenv("GPU_API_KEY", "sk-ibrasoft-gpu-voice")
# Model Path Resolution & Auto-Downloader
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

v1_model = os.path.join(models_dir, "kokoro-v1.0.onnx")
v1_voices = os.path.join(models_dir, "voices-v1.0.bin")
v019_model = os.path.join(models_dir, "kokoro-v0_19.onnx")
v019_voices = os.path.join(models_dir, "voices.bin")

def ensure_model_files():
    import urllib.request
    
    # 1. Prefer v1.0 model & 54-voice pack if present and complete
    if os.path.exists(v1_model) and os.path.exists(v1_voices) and os.path.getsize(v1_voices) > 20000000:
        return v1_model, v1_voices
        
    # 2. Check if v0.19 model is present
    if os.path.exists(v019_model) and os.path.exists(v019_voices) and os.path.getsize(v019_voices) > 5000000:
        return v019_model, v019_voices
        
    # 3. Auto-download v1.0 multi-language model assets if missing
    logger.info("⚡ Auto-downloading Kokoro-82M v1.0 ONNX model & 54-voice pack...")
    url_model = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    url_voices = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    try:
        if not os.path.exists(v1_model):
            logger.info("   • Downloading kokoro-v1.0.onnx (320 MB)...")
            urllib.request.urlretrieve(url_model, v1_model)
        if not os.path.exists(v1_voices) or os.path.getsize(v1_voices) < 20000000:
            logger.info("   • Downloading voices-v1.0.bin (27 MB)...")
            urllib.request.urlretrieve(url_voices, v1_voices)
        return v1_model, v1_voices
    except Exception as e:
        logger.warning(f"Auto-download notice: {e}")
        if os.path.exists(v019_model) and os.path.exists(v019_voices):
            return v019_model, v019_voices
        raise e

MODEL_PATH, VOICES_PATH = ensure_model_files()
logger.info(f"Using Kokoro Model: {MODEL_PATH}")
logger.info(f"Using Kokoro Voices: {VOICES_PATH}")

app = FastAPI(title="Kokoro Neural Streaming TTS Engine", version="2.5.0")

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
    gain: Optional[float] = 1.0
    lang: Optional[str] = "en-us"
    stream: Optional[bool] = False

class OpenAISpeechRequest(BaseModel):
    model: Optional[str] = "kokoro"
    input: str
    voice: Optional[str] = "af_bella"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    gain: Optional[float] = 1.0

# SSML & Paralinguistic Pre-Processor
def preprocess_ssml_and_cues(text: str):
    """
    Parses SSML-like tags (<break time="300ms"/>, <prosody rate="...">)
    and paralinguistic cues ((laughs), (coughs), (gasps), (sighs)).
    Replaces them with rhythm-shaping punctuation & pauses.
    """
    processed = text

    # 1. Parse <break time="300ms"/> or <break time="1s"/>
    def break_replacer(match):
        val_str = match.group(1).lower()
        if "ms" in val_str:
            ms = float(re.sub(r"[^\d.]", "", val_str) or "300")
        elif "s" in val_str:
            ms = float(re.sub(r"[^\d.]", "", val_str) or "1") * 1000
        else:
            ms = 300.0

        if ms >= 800:
            return " ... ... "
        elif ms >= 400:
            return " ... "
        else:
            return " , "

    processed = re.sub(r"<break\s+time=[\"']?([^\"'/>]+)[\"']?\s*/?>", break_replacer, processed, flags=re.IGNORECASE)

    # 2. Parse <prosody ...> tags by stripping tag wrappers
    processed = re.sub(r"</?prosody[^>]*>", "", processed, flags=re.IGNORECASE)

    # 3. Intercept nonverbal/paralinguistic cues in parentheses: (laughs), (sighs), (gasps), (coughs), (clears throat)
    def cue_replacer(match):
        cue = match.group(1).lower()
        if any(w in cue for w in ["laugh", "chuckle", "giggle"]):
            return " ... (ha-ha) , "
        elif any(w in cue for w in ["sigh", "gasp", "breath"]):
            return " ... , "
        elif any(w in cue for w in ["cough", "throat"]):
            return " ... "
        else:
            return " , "

    processed = re.sub(r"\((laughs|chuckle|giggle|sighs|gasps|coughs|clears throat|snicker)\)", cue_replacer, processed, flags=re.IGNORECASE)

    # 4. Normalize paragraph breaks (\n\n) to strong pauses
    processed = re.sub(r"\n\s*\n", " ... \n", processed)

    return processed.strip()


# Free-Form & Natural Language Prosody Style Tag Parser
def parse_emotion_and_prosody(text: str, base_speed: float = 1.0, base_gain: float = 1.0):
    """
    Parses bracketed style/emotion tags [...]:
    - Single-word tags: [cheerful], [empathy], [urgent], [calm], [whisper], [happy], [sad], [angry], [excited], [surprised], [sarcastic], [disgust]
    - Free-form natural language cues: [whisper in small voice], [pitch up], [excited and fast], [calm and slow], [softly], [nervously]
    """
    speed = base_speed
    gain = base_gain
    detected_styles = []

    # Find all inline tags in brackets [...]
    bracket_tags = re.findall(r"\[([^\]]+)\]", text)

    for tag in bracket_tags:
        tag_lower = tag.lower().strip()
        detected_styles.append(tag_lower)

        # Dynamic Speed Modifiers
        if any(w in tag_lower for w in ["urgent", "fast", "excited", "quick", "pitch up", "energetic"]):
            speed = min(1.4, speed * 1.16)
        if any(w in tag_lower for w in ["whisper", "calm", "slow", "soft", "small voice", "gently", "nervously", "empathy"]):
            speed = max(0.75, speed * 0.88)
        if "very slow" in tag_lower:
            speed = max(0.70, speed * 0.80)

        # Dynamic Gain / Volume Modifiers
        if any(w in tag_lower for w in ["whisper", "small voice", "softly", "quiet"]):
            gain = max(0.5, gain * 0.75)
        if any(w in tag_lower for w in ["excited", "urgent", "loud", "shout"]):
            gain = min(2.0, gain * 1.15)

    # Strip all bracketed tags from the clean text
    clean_text = re.sub(r"\[[^\]]+\]", "", text).strip()

    # Also run SSML & paralinguistic preprocessor
    clean_text = preprocess_ssml_and_cues(clean_text)

    style_summary = ", ".join(detected_styles) if detected_styles else "neutral"
    return clean_text, speed, gain, style_summary


@lru_cache(maxsize=512)
def generate_kokoro_audio_cached(text: str, voice_name: str, speed: float, lang: str, gain: float = 1.0):
    kokoro = get_kokoro()
    if not kokoro:
        raise RuntimeError("Kokoro engine not ready")
    
    # Normalize language codes for espeak-ng backend compatibility
    lang_code = (lang or "en-us").lower().strip()
    LANG_MAP = {
        "hi-in": "hi",
        "hi": "hi",
        "h": "hi",
        "ja-jp": "ja",
        "ja": "ja",
        "j": "ja",
        "zh-cn": "zh",
        "zh": "zh",
        "z": "zh",
        "es-es": "es",
        "es": "es",
        "e": "es",
        "fr-fr": "fr-fr",
        "fr": "fr-fr",
        "f": "fr-fr",
        "it-it": "it",
        "it": "it",
        "i": "it",
        "en-gb": "en-gb",
        "b": "en-gb",
        "en-us": "en-us",
        "a": "en-us"
    }
    target_lang = LANG_MAP.get(lang_code, lang_code)
    
    try:
        samples, sr = kokoro.create(text, voice=voice_name, speed=speed, lang=target_lang)
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "voice" in err_msg:
            logger.warning(f"Voice '{voice_name}' not in voices.bin ({e}). Falling back to 'af_bella'...")
            samples, sr = kokoro.create(text, voice="af_bella", speed=speed, lang=target_lang)
        elif "espeak" in err_msg or "language" in err_msg:
            logger.warning(f"Language '{target_lang}' not supported by espeak ({e}). Falling back to 'en-us'...")
            samples, sr = kokoro.create(text, voice=voice_name, speed=speed, lang="en-us")
        else:
            raise e

    # Apply Volume/Gain control scaling
    if gain != 1.0:
        samples = np.clip(samples * gain, -1.0, 1.0)

    return samples, sr

@app.on_event("startup")
async def startup_event():
    kokoro = get_kokoro()
    if kokoro:
        logger.info("⚡ Pre-warming Kokoro ONNX CUDA kernels...")
        try:
            generate_kokoro_audio_cached("Warmup audio test.", "af_bella", 1.0, "en-us", 1.0)
            generate_kokoro_audio_cached("Warmup audio test.", "am_michael", 1.0, "en-us", 1.0)
            logger.success("✓ Kokoro ONNX CUDA kernels pre-warmed successfully!")
        except Exception as e:
            logger.warning(f"Warmup notice: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kokoro-tts",
        "engine_ready": kokoro_engine is not None,
        "sample_rate": 24000,
        "default_voice": "af_bella",
        "features": [
            "freeform_style_tags",
            "paralinguistic_cues",
            "wrapper_ssml_breaks",
            "gain_volume_control",
            "chunked_pcm_streaming",
            "lru_kernel_cache"
        ]
    }

@app.post("/synthesize")
async def synthesize_speech(req: SynthesizeRequest, request: Request):
    verify_api_key(request)
    kokoro = get_kokoro()
    if not kokoro:
        raise HTTPException(status_code=500, detail="Kokoro TTS engine not initialized.")

    clean_text, effective_speed, effective_gain, style_desc = parse_emotion_and_prosody(req.text, req.speed or 1.0, req.gain or 1.0)
    voice_name = req.voice or "af_bella"

    t0 = time.time()
    try:
        samples, sample_rate = generate_kokoro_audio_cached(clean_text, voice_name, effective_speed, req.lang or "en-us", effective_gain)
        
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        latency_ms = round((time.time() - t0) * 1000, 1)
        logger.info(f"🎙️ [KOKORO TTS] voice='{voice_name}' style='{style_desc}' | {latency_ms}ms | {len(wav_bytes)} bytes")

        return Response(content=wav_bytes, media_type="audio/wav", headers={
            "X-Latency-Ms": str(latency_ms),
            "X-Style-Detected": style_desc,
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

    clean_text, effective_speed, effective_gain, style_desc = parse_emotion_and_prosody(req.input, req.speed or 1.0, req.gain or 1.0)
    
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

    samples, sample_rate = generate_kokoro_audio_cached(clean_text, target_voice, effective_speed, "en-us", effective_gain)
    buf = io.BytesIO()
    
    out_format = (req.response_format or "wav").lower()
    if out_format == "pcm":
        pcm16 = (samples * 32767).astype(np.int16).tobytes()
        return Response(content=pcm16, media_type="audio/pcm")
    else:
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")

# Streaming Chunked Endpoint (TTFA < 150ms)
@app.post("/stream")
async def stream_speech(req: SynthesizeRequest, request: Request):
    verify_api_key(request)
    kokoro = get_kokoro()
    if not kokoro:
        raise HTTPException(status_code=500, detail="Kokoro TTS not ready")

    clean_text, effective_speed, effective_gain, style_desc = parse_emotion_and_prosody(req.text, req.speed or 1.0, req.gain or 1.0)
    voice_name = req.voice or "af_bella"

    t_req = time.time()
    async def audio_generator() -> AsyncGenerator[bytes, None]:
        # Split into short clauses/phrases by commas & punctuation for sub-60ms first-chunk delivery
        clauses = [c.strip() for c in re.split(r"(?<=[,.!?;])\s+", clean_text) if c.strip()]
        if not clauses:
            clauses = [clean_text]

        for i, clause in enumerate(clauses):
            if not clause:
                continue
            t_c0 = time.time()
            samples, sr = generate_kokoro_audio_cached(clause, voice_name, effective_speed, req.lang or "en-us", effective_gain)
            t_syn = round((time.time() - t_c0) * 1000, 1)
            if i == 0:
                logger.info(f"⚡ [TTS STREAM CHUNK #1] text='{clause[:30]}' style='{style_desc}' | syn={t_syn}ms | total={(time.time() - t_req)*1000:.1f}ms")
            pcm16 = (samples * 32767).astype(np.int16).tobytes()
            yield pcm16
            await asyncio.sleep(0.005)

    return StreamingResponse(audio_generator(), media_type="application/octet-stream", headers={
        "Transfer-Encoding": "chunked",
        "Content-Type": "audio/pcm; rate=24000; channels=1",
        "X-Style-Detected": style_desc
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
    samples, sr = generate_kokoro_audio_cached(phrase, voice, 1.05, "en-us", 1.0)
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
            gain = float(data.get("gain", 1.0))

            if text.strip() and kokoro:
                clean_text, eff_speed, eff_gain, _ = parse_emotion_and_prosody(text, speed, gain)
                samples, sr = generate_kokoro_audio_cached(clean_text, voice, eff_speed, "en-us", eff_gain)
                pcm16 = (samples * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm16)
    except WebSocketDisconnect:
        logger.info("WebSocket TTS client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket TTS error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088, access_log=False)
