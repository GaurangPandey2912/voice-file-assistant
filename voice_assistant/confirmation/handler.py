from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyttsx3

logger = logging.getLogger("voice_assistant.confirmation")


class ConfirmationHandler:
    def __init__(self, speak: bool = True) -> None:
        self._speak_enabled = speak
        self._tts: Optional[pyttsx3.Engine] = None

    def _get_tts(self) -> pyttsx3.Engine:
        if self._tts is None:
            try:
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 160)
                self._tts.setProperty("volume", 0.8)
            except Exception as e:
                logger.warning(f"TTS initialization failed: {e}")
                self._speak_enabled = False
        return self._tts

    def speak(self, message: str) -> None:
        if self._speak_enabled:
            try:
                tts = self._get_tts()
                tts.say(message)
                tts.runAndWait()
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

    def confirm_destructive(self, action: str, target: str) -> bool:
        prompt = f"{action} {target}?"
        print(f"\n  Confirm: {prompt} (yes/no): ", end="", flush=True)
        try:
            response = input().strip().lower()
            confirmed = response in ("yes", "y", "yeah", "sure", "ok", "okay")
            if confirmed:
                logger.info(f"Confirmed: {prompt}")
            else:
                logger.info(f"Cancelled: {prompt}")
            return confirmed
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    def confirm_interpretation(self, description: str) -> bool:
        """Used by the confidence-based decision layer: when the intent parser
        wasn't confident about how it interpreted a command (e.g. it fell back
        to the spaCy heuristic parser rather than a direct pattern match), ask
        the user to confirm the interpretation before executing anything."""
        prompt = f"Did you mean: {description}?"
        print(f"\n  Confirm: {prompt} (yes/no): ", end="", flush=True)
        try:
            response = input().strip().lower()
            confirmed = response in ("yes", "y", "yeah", "sure", "ok", "okay")
            if confirmed:
                logger.info(f"Interpretation confirmed: {prompt}")
            else:
                logger.info(f"Interpretation rejected: {prompt}")
            return confirmed
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    def resolve_ambiguous(
        self,
        matches: list[Path],
        action_hint: str = "",
    ) -> Optional[Path]:
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        print(f"\n  Multiple files found {'for ' + action_hint if action_hint else ''}:")
        for i, match in enumerate(matches, 1):
            extra = ""
            if match.is_dir():
                extra = " [folder]"
            elif match.suffix:
                extra = f" [{match.suffix}]"
            print(f"  {i}. {match.name}{extra}  ({match.parent.name})")

        print("  Please enter the number or filename: ", end="", flush=True)
        try:
            response = input().strip()
            if response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]
            else:
                for match in matches:
                    if match.name.lower() == response.lower():
                        return match
                    if match.stem.lower() == response.lower():
                        return match
            print(f"  Invalid selection: {response}")
            return None
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def validate_path(self, path: Path, must_exist: bool = True) -> Optional[str]:
        if must_exist and not path.exists():
            return f"Path does not exist: {path}"
        if path.exists() and not must_exist:
            return f"Path already exists: {path}"
        try:
            resolved = path.resolve()
            if must_exist and not resolved.exists():
                return f"Path cannot be resolved: {path}"
        except (OSError, RuntimeError) as e:
            return f"Path validation error: {e}"
        return None
