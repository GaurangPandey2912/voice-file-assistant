from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import whisper

logger = logging.getLogger("voice_assistant.speech")


class SpeechTranscriber:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model: Optional[whisper.Whisper] = None

    def _load_model(self) -> None:
        if self._model is None:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        if audio is None or len(audio) == 0:
            return None
        try:
            self._load_model()
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            result = self._model.transcribe(audio, language="en")
            text = result.get("text", "").strip()
            if text:
                logger.debug(f"Transcribed: '{text}'")
                return text
            return None
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
