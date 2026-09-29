from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from pynput import keyboard

import logging

logger = logging.getLogger("voice_assistant.audio")

_HOTKEY_MAP: dict[str, keyboard.Key] = {
    "ctrl_l": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl": keyboard.Key.ctrl_l,
    "alt_l": keyboard.Key.alt_l,
    "alt_r": keyboard.Key.alt_r,
    "alt": keyboard.Key.alt_l,
    "cmd_l": keyboard.Key.cmd_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd": keyboard.Key.cmd_l,
    "shift_l": keyboard.Key.shift_l,
    "shift_r": keyboard.Key.shift_r,
    "shift": keyboard.Key.shift_l,
    "tab": keyboard.Key.tab,
    "space": keyboard.Key.space,
    "enter": keyboard.Key.enter,
    "esc": keyboard.Key.esc,
    "caps_lock": keyboard.Key.caps_lock,
}


def _parse_hotkey(hotkey_str: str) -> keyboard.Key:
    match = re.search(r"<([^>]+)>", hotkey_str)
    if match:
        key_name = match.group(1)
        if key_name in _HOTKEY_MAP:
            return _HOTKEY_MAP[key_name]
        logger.warning(f"Unknown hotkey '{key_name}', falling back to Left Ctrl")
    return keyboard.Key.ctrl_l


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._recording: list[np.ndarray] = []
        self._is_recording = False
        self._stream: Optional[sd.InputStream] = None
        self._hotkey_pressed = False
        self._hotkey_listener: Optional[keyboard.Listener] = None
        self._on_audio_ready: Optional[Callable[[np.ndarray], None]] = None
        self._hotkey_key: keyboard.Key = keyboard.Key.ctrl_l
        self._lock = threading.Lock()

    def start_recording(self) -> None:
        with self._lock:
            if self._is_recording:
                return
            self._recording = []
            self._is_recording = True

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning(f"Audio input status: {status}")
            if self._is_recording:
                self._recording.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        logger.debug("Recording started")

    def stop_recording(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._recording:
            logger.debug("No audio captured")
            return None

        audio_data = np.concatenate(self._recording, axis=0).flatten()
        self._recording = []
        logger.debug(f"Recording stopped, {len(audio_data)} samples captured")
        return audio_data

    def _on_press(self, key) -> None:
        try:
            if key == self._hotkey_key:
                if not self._hotkey_pressed:
                    self._hotkey_pressed = True
                    self.start_recording()
        except AttributeError:
            pass

    def _on_release(self, key) -> None:
        try:
            if key == self._hotkey_key:
                if self._hotkey_pressed:
                    self._hotkey_pressed = False
                    audio = self.stop_recording()
                    if audio is not None and self._on_audio_ready:
                        self._on_audio_ready(audio)
        except AttributeError:
            pass

    def start_hotkey_listener(
        self,
        on_audio_ready: Callable[[np.ndarray], None],
        hotkey_str: str = "<ctrl>+<lc>",
    ) -> None:
        self._on_audio_ready = on_audio_ready
        self._hotkey_key = _parse_hotkey(hotkey_str)
        self._hotkey_listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._hotkey_listener.start()
        key_name = str(self._hotkey_key).replace("Key.", "").replace("_", " ").title()
        logger.info(f"Hotkey listener started ({key_name}: push-to-talk)")

    def stop_hotkey_listener(self) -> None:
        if self._hotkey_listener:
            self._hotkey_listener.stop()
            self._hotkey_listener = None
        self.stop_recording()
        logger.info("Hotkey listener stopped")
