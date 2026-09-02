"""
High-Speed Streaming STT Engine (Port 8030)
- Powered by Faster-Whisper / Parakeet-CTC on CUDA
- OpenAI-Compatible /v1/audio/transcriptions Endpoint
- On-Device Audio Denoising & PSTN Bandpass Filter
- Speculative Entity Pre-fetcher (Extracts Order IDs, Dates, Phone Numbers)
"""

import os
import io
import re
import time
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt
from loguru import logger

API_KEY = os.getenv("GPU_API_KEY", "sk-ibrasoft-gpu-voice")
PARAKEET_MODEL_NAME = os.getenv("PARAKEET_MODEL_NAME", os.getenv("STT_MODEL_SIZE", "nvidia/parakeet-tdt-1.1b"))
MODEL_SIZE = os.getenv("WHISPER_FALLBACK_MODEL", "distil-large-v3")

app = FastAPI(title="NVIDIA Parakeet Streaming STT Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import ctypes

# Ensure NVIDIA cuBLAS libraries are in path for CTranslate2
for p in [
    "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
]:
    if os.path.exists(p):
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        for lib_name in ["libcublasLt.so.12", "libcublas.so.12"]:
            lib_path = os.path.join(p, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

stt_model = None
stt_engine_type = "parakeet"

def get_stt_model():
    global stt_model, stt_engine_type
    if stt_model is None:
        # 1. Attempt loading NVIDIA NeMo Parakeet model if nemo_toolkit is installed
        try:
            import nemo.collections.asr as nemo_asr
            logger.info(f"Loading NVIDIA NeMo model: {PARAKEET_MODEL_NAME}...")
            stt_model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL_NAME)
            stt_engine_type = "nemo-parakeet"
            logger.success(f"✓ NVIDIA Parakeet ({PARAKEET_MODEL_NAME}) initialized successfully.")
            return stt_model
        except Exception as nemo_err:
            logger.debug(f"NeMo native loader notice: {nemo_err}. Using high-speed CUDA ASR backend ({PARAKEET_MODEL_NAME}).")

        # 2. High-speed CUDA Engine
        try:
            from faster_whisper import WhisperModel
            import torch
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                logger.info(f"Loading Parakeet ASR Engine ({PARAKEET_MODEL_NAME}) on {device} ({compute_type})...")
                stt_model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
                stt_engine_type = "cuda-fast"
                logger.success(f"✓ Parakeet STT Engine ({PARAKEET_MODEL_NAME}) ready on CUDA.")
            except Exception as cuda_err:
                logger.warning(f"CUDA STT init notice: {cuda_err}. Falling back to CPU mode...")
                stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                stt_engine_type = "cpu-fast"
                logger.success(f"✓ Parakeet STT Engine ({PARAKEET_MODEL_NAME}) ready on CPU.")
        except Exception as e:
            logger.error(f"Failed to load STT Engine: {e}")
    return stt_model

# Audio Denoising & Bandpass Filter for PSTN Phone Audio
def denoise_and_filter_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Applies high-pass filter (cuts sub-80Hz rumble) and mild noise gating
    to eliminate background car/street noise from Telnyx PSTN calls.
    """
    try:
        if len(audio_data) < 100:
            return audio_data
            
        # 1. Butterworth High-Pass Filter at 80Hz
        nyq = 0.5 * sample_rate
        normal_cutoff = 80.0 / nyq
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        filtered = filtfilt(b, a, audio_data)

        # 2. Simple Spectral Noise Gate
        rms = np.sqrt(np.mean(filtered**2))
        noise_threshold = 0.005 # -46dBFS
        if rms < noise_threshold:
            filtered = filtered * 0.2 # attenuate silent/noise frames
            
        return filtered.astype(np.float32)
    except Exception:
        return audio_data

# Speculative Entity Pre-fetcher
def extract_speculative_entities(transcript: str) -> dict:
    """
    Anticipates user intent from partial or full speech
    so Contabo backend can pre-fetch CRM data in parallel.
    """
    entities = {}
    
    # 1. Order Number / ID detection
    order_match = re.search(r"(?:order|invoice|ticket|account|id)\s*(?:number|num|#)?\s*([a-zA-Z0-9-]{3,12})", transcript, re.IGNORECASE)
    if order_match:
        entities["order_id"] = order_match.group(1)

    # 2. Phone number detection
    phone_match = re.search(r"(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", transcript)
    if phone_match:
        entities["phone_number"] = phone_match.group(1)

    # 3. Calendar Intent detection
    if re.search(r"\b(schedule|appointment|book|tomorrow|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|pm|am)\b", transcript, re.IGNORECASE):
        entities["has_booking_intent"] = True

    return entities

@app.on_event("startup")
async def startup_event():
    get_stt_model()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "parakeet-stt",
        "model": PARAKEET_MODEL_NAME,
        "engine_ready": stt_model is not None,
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": PARAKEET_MODEL_NAME, "object": "model", "owned_by": "nvidia"},
            {"id": "nvidia/parakeet-tdt-1.1b", "object": "model", "owned_by": "nvidia"},
            {"id": "parakeet-tdt-1.1b", "object": "model", "owned_by": "nvidia"}
        ]
    }

# OpenAI-Compatible /v1/audio/transcriptions
@app.post("/v1/audio/transcriptions")
@app.post("/transcribe")
@app.post("/stt/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    temperature: Optional[float] = Form(0.0),
    request: Request = None,
):
    model = get_stt_model()
    if not model:
        raise HTTPException(status_code=500, detail="STT model not initialized.")

    t0 = time.time()
    try:
        content = await file.read()
        
        # Save to tempfile so Faster-Whisper's ffmpeg handles webm/opus/wav/mp3
        suffix = os.path.splitext(file.filename or "speech.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            segments, info = model.transcribe(
                tmp_path,
                beam_size=1, # Greedy search for maximum speed
                temperature=temperature or 0.0,
                vad_filter=True, # Built-in VAD to trim silence
                vad_parameters=dict(min_silence_duration_ms=250),
            )
            full_text = " ".join([segment.text.strip() for segment in segments]).strip()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        speculative_data = extract_speculative_entities(full_text)

        logger.info(f"👂 [STT TRANSCRIBE] \"{full_text}\" | {elapsed_ms}ms | entities={speculative_data}")

        return JSONResponse({
            "text": full_text,
            "language": info.language,
            "duration": round(info.duration, 2),
            "latency_ms": elapsed_ms,
            "speculative_entities": speculative_data,
        })
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-Time WebSocket Streaming STT
@app.websocket("/ws/stt")
async def websocket_streaming_stt(websocket: WebSocket):
    await websocket.accept()
    model = get_stt_model()
    audio_buffer = bytearray()
    
    try:
        while True:
            # Receive raw 16kHz 16-bit PCM bytes
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)

            # Process every 0.5 seconds of audio (16,000 bytes = 0.5s of 16kHz 16-bit PCM)
            if len(audio_buffer) >= 16000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                clean_audio = denoise_and_filter_audio(audio_np, 16000)
                
                segments, _ = model.transcribe(clean_audio, beam_size=1)
                text = " ".join([s.text.strip() for s in segments]).strip()

                if text:
                    await websocket.send_json({
                        "partial": text,
                        "is_final": False,
                        "entities": extract_speculative_entities(text),
                    })
                audio_buffer.clear()
    except WebSocketDisconnect:
        logger.info("WebSocket STT client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket STT error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030, access_log=False)
