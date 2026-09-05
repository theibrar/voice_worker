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
MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "nvidia/parakeet-tdt-1.1b")

app = FastAPI(title="NVIDIA Parakeet-TDT v3 GPU Streaming STT Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import ctypes
import site
import glob

# Ensure NVIDIA cuBLAS libraries are in path for CTranslate2 / NeMo across all Python versions
nvidia_search_paths = []
for base in site.getsitepackages() + [site.getusersitepackages(), "/usr/local/lib", "/usr/lib"]:
    if os.path.exists(base):
        for pattern in ["**/nvidia/cublas/lib", "**/nvidia/cudnn/lib", "**/nvidia/cuda_runtime/lib"]:
            for match in glob.glob(os.path.join(base, pattern), recursive=True):
                if match not in nvidia_search_paths:
                    nvidia_search_paths.append(match)

for py_ver in ["python3.10", "python3.11", "python3.12"]:
    for sub in ["cublas", "cudnn", "cuda_runtime"]:
        p = f"/usr/local/lib/{py_ver}/dist-packages/nvidia/{sub}/lib"
        if os.path.exists(p) and p not in nvidia_search_paths:
            nvidia_search_paths.append(p)

for p in nvidia_search_paths:
    if p not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for lib_name in ["libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"]:
        lib_path = os.path.join(p, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

stt_model = None

def ensure_peft_installed():
    try:
        import peft
    except ImportError:
        logger.info("⚡ Auto-installing missing NeMo dependency 'peft'...")
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "peft", "transformers", "--no-cache-dir"])
        except Exception as peft_err:
            logger.warning(f"Could not auto-install peft: {peft_err}")

def get_stt_model():
    global stt_model
    if stt_model is None:
        model_name = os.getenv("STT_MODEL_SIZE", "nvidia/parakeet-tdt-1.1b")
        logger.info(f"Loading NVIDIA Parakeet-TDT (v3) ASR Engine ({model_name})...")
        ensure_peft_installed()
        try:
            import torch
            import nemo.collections.asr as nemo_asr
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(nemo_asr.models, "ASRModel"):
                stt_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            elif hasattr(nemo_asr.models, "EncDecRNNTBModel"):
                stt_model = nemo_asr.models.EncDecRNNTBModel.from_pretrained(model_name=model_name)
            elif hasattr(nemo_asr.models, "EncDecCTCModelBPE"):
                stt_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
            else:
                stt_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name)

            if device == "cuda" and hasattr(stt_model, "cuda"):
                stt_model = stt_model.cuda()
            if hasattr(stt_model, "eval"):
                stt_model.eval()
            logger.success(f"✓ NVIDIA Parakeet-TDT (v3) ASR Engine ({model_name}) initialized on {device.upper()}.")
        except Exception as e:
            logger.warning(f"NeMo Parakeet-TDT init notice ({e}). Loading via Faster-Whisper runner...")
            try:
                from faster_whisper import WhisperModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                stt_model = WhisperModel("distil-large-v3", device=device, compute_type=compute_type)
                logger.success("✓ Faster-Whisper ASR Engine initialized.")
            except Exception as e2:
                logger.error(f"Failed to load Parakeet-TDT STT Engine: {e2}")
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
        "service": "streaming-stt",
        "model": MODEL_SIZE,
        "engine_ready": stt_model is not None,
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
        
        suffix = os.path.splitext(file.filename or "speech.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            full_text = ""
            detected_lang = language or "en"
            
            # NeMo Parakeet-TDT ASR transcribe method
            if hasattr(model, "transcribe") and not hasattr(model, "model"):
                try:
                    res = model.transcribe([tmp_path])
                    if isinstance(res, list) and len(res) > 0:
                        full_text = str(res[0]).strip()
                    else:
                        full_text = str(res).strip()
                except Exception as nemo_err:
                    logger.warning(f"NeMo transcribe notice: {nemo_err}, using fallback...")
                    full_text = ""
            
            # Faster-Whisper ASR transcribe fallback
            if not full_text and hasattr(model, "transcribe"):
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=1,
                    temperature=temperature or 0.0,
                )
                full_text = " ".join([segment.text.strip() for segment in segments]).strip()
                if hasattr(info, "language"):
                    detected_lang = info.language
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
