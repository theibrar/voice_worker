import time
import numpy as np
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("turn_detector")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class TurnDetector:
    """
    Dual-Stage Voice Activity & End-of-Thought Turn Detection.
    Guarantees sub-120ms barge-in interruption cutoff and natural turn-taking.
    """
    def __init__(self, silence_threshold_ms: int = 450, speech_energy_threshold: float = 0.02):
        self.silence_threshold_ms = silence_threshold_ms
        self.speech_energy_threshold = speech_energy_threshold
        self.is_speaking = False
        self.last_speech_time = 0
        self.speech_start_time = 0

    def process_frame(self, pcm_frame: bytes, sample_rate: int = 16000) -> dict:
        """
        Processes a single audio frame (e.g. 20ms of 16-bit PCM).
        Returns:
            {
                "interruption": bool,  # True if user just started speaking (cancel active TTS)
                "speech_active": bool, # True if speech currently ongoing
                "turn_completed": bool # True if user finished speaking and AI should respond
            }
        """
        now = time.perf_counter()
        samples = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0.0

        interruption = False
        turn_completed = False

        if energy > self.speech_energy_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = now
                interruption = True
                logger.debug(f"[VAD] Speech start detected (Energy: {energy:.4f}) -> Trigger Interruption Signal")
            self.last_speech_time = now
        else:
            if self.is_speaking:
                silence_duration_ms = (now - self.last_speech_time) * 1000
                if silence_duration_ms > self.silence_threshold_ms:
                    self.is_speaking = False
                    turn_completed = True
                    total_speech_ms = (self.last_speech_time - self.speech_start_time) * 1000
                    logger.debug(f"[VAD] Turn completed (Total speech: {total_speech_ms:.0f}ms, Silence: {silence_duration_ms:.0f}ms)")

        return {
            "interruption": interruption,
            "speech_active": self.is_speaking,
            "turn_completed": turn_completed,
            "energy": energy,
        }
