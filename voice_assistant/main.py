from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional
import re
import numpy as np

from voice_assistant.audio.recorder import AudioRecorder
from voice_assistant.config.config_manager import ConfigManager
from voice_assistant.confirmation.handler import ConfirmationHandler
from voice_assistant.filesystem.indexer import FileIndexer
from voice_assistant.filesystem.operations import (
    create_file,
    create_folder,
    delete,
    duplicate_file,
    rename,
    move,
    copy_op,
    open_file,
    navigate,
    list_items,
    resolve_path,
    resolve_ambiguous,
    get_current_dir,
    set_current_dir,
    known_folder,
)
from voice_assistant.intent.parser import IntentParser, IntentType, ParsedIntent
from voice_assistant.logs.logger import setup_logger
from voice_assistant.search.engine import SearchEngine
from voice_assistant.speech.transcriber import SpeechTranscriber
from voice_assistant.ui.feedback import UIFeedback

logger = logging.getLogger("voice_assistant")

# Recent-interaction context window: how many recently touched files are kept
# and offered to the search engine as a ranking signal.
_RECENT_HISTORY_SIZE = 10


_HOTKEY_LABELS: dict[str, str] = {
    "ctrl_l": "Left Ctrl",
    "ctrl_r": "Right Ctrl",
    "ctrl": "Left Ctrl",
    "alt_l": "Left Option",
    "alt_r": "Right Option",
    "alt": "Left Option",
    "cmd_l": "Left Cmd",
    "cmd_r": "Right Cmd",
    "cmd": "Left Cmd",
    "shift_l": "Left Shift",
    "shift_r": "Right Shift",
    "shift": "Left Shift",
}


def _hotkey_label(hotkey_str: str) -> str:
    import re
    match = re.search(r"<([^>]+)>", hotkey_str)
    if match:
        key_name = match.group(1)
        if key_name in _HOTKEY_LABELS:
            return _HOTKEY_LABELS[key_name]
    return "Left Ctrl"


def _resolve_destination(dest: str) -> Optional[Path]:
    known = {
        "desktop": known_folder("desktop"),
        "documents": known_folder("documents"),
        "downloads": known_folder("downloads"),
        "projects": known_folder("projects"),
        "home": Path.home(),
    }
    key = dest.strip().lower()
    if key.startswith("the "):
        key = key[4:]
    if key.endswith(" folder"):
        key = key[: -len(" folder")]
    elif key.endswith(" directory"):
        key = key[: -len(" directory")]
    if key in known:
        return known[key]
    candidate = Path(dest.strip())
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for name in (key, dest.strip()):
        for loc in [get_current_dir(), *known.values()]:
            cand = loc / name
            if cand.exists():
                return cand
    return None


def _describe_intent(intent: ParsedIntent) -> str:
    """Human-readable summary of a parsed intent, used when the confidence-based
    decision layer needs to ask the user to confirm an uncertain interpretation."""
    verb = {
        IntentType.CREATE_FILE: "create a file",
        IntentType.CREATE_FOLDER: "create a folder",
        IntentType.DELETE: "delete",
        IntentType.DUPLICATE: "duplicate",
        IntentType.RENAME: "rename",
        IntentType.MOVE: "move",
        IntentType.COPY: "copy",
        IntentType.OPEN: "open",
        IntentType.NAVIGATE: "navigate to",
        IntentType.LIST: "list items",
        IntentType.SEARCH: "search for",
    }.get(intent.intent, "run that command")

    target = intent.filename or intent.source or ""
    parts = [verb]
    if target:
        parts.append(f"'{target}'")
    if intent.destination:
        parts.append(f"into '{intent.destination}'")
    return " ".join(parts)


class VoiceFileAssistant:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get()
        self.logger = setup_logger(
            Path(self.config.log_dir), name="voice_assistant"
        )
        hotkey_label = _hotkey_label(self.config.hotkey)
        self.ui = UIFeedback(hotkey_label=hotkey_label)
        self.confirmer = ConfirmationHandler(speak=True)
        self.recorder = AudioRecorder()
        self.transcriber = SpeechTranscriber(self.config.whisper_model)
        self.parser = IntentParser()
        self.indexer = FileIndexer(self.config.index_dir)
        self.search_engine = SearchEngine([])
        self._running = False
        # Recent-interaction context: most-recently-touched file first.
        self._recent_paths: deque[Path] = deque(maxlen=_RECENT_HISTORY_SIZE)

    def _remember(self, path: Optional[Path]) -> None:
        if path is None:
            return
        path = Path(path)
        # Keep only the most recent occurrence of a given file.
        if path in self._recent_paths:
            self._recent_paths.remove(path)
        self._recent_paths.appendleft(path)

    def initialize(self) -> None:
        self.ui.header()

        self.indexer.set_watched_folders(self.config.indexed_folders)
        index_data = self.indexer.get_index()
        if not index_data:
            self.logger.info("No existing index found, building...")
            index_data = self.indexer.build_index()
        self.search_engine.update_index(index_data)
        self.indexer.start_watching()
        self.logger.info(
            f"Index loaded: {len(index_data)} items across "
            f"{len(self.config.indexed_folders)} folders"
        )

        self.recorder.start_hotkey_listener(
            self._on_audio_ready, hotkey_str=self.config.hotkey
        )
        self._running = True

    def _on_audio_ready(self, audio: np.ndarray) -> None:
        try:
            self._process_pipeline(audio)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            self.ui.show_error(str(e))
            self.confirmer.speak("An error occurred. Please try again.")

    def _process_pipeline(self, audio: np.ndarray) -> None:
        start_time = time.time()
        current_dir = get_current_dir()
        self.ui.show_current_dir(current_dir)

        text = self.transcriber.transcribe(audio)
        if not text:
            self.ui.show_error("Could not understand audio. Please try again.")
            self.confirmer.speak("I could not understand. Please try again.")
            return

        self.ui.show_speech(text)

        intent = self.parser.parse(text)
        self.ui.show_intent(intent)

        if intent.intent == IntentType.UNKNOWN:
            msg = f"Could not understand command: {text}"
            self.ui.show_error(msg)
            self.confirmer.speak("Command not recognized. Please try again.")
            return

        # Confidence-based decision layer: a confident regex match executes
        # directly; an uncertain (spaCy fallback / keyword) interpretation is
        # confirmed with the user first rather than risking the wrong action.
        if intent.is_low_confidence:
            description = _describe_intent(intent)
            self.ui.show_error(  # reuse as a neutral inline notice, not a failure
                f"Low-confidence interpretation ({intent.parse_method}): {description}"
            )
            if not self.confirmer.confirm_interpretation(description):
                self.ui.show_result(False, "Cancelled — interpretation not confirmed")
                return

        success, message = self._execute_intent(intent)

        self.ui.show_result(success, message)
        if not success:
            self.confirmer.speak(message)

        latency = time.time() - start_time
        self.logger.info(
            f"Pipeline completed",
            extra={
                "speech": text,
                "intent": intent.intent.name,
                "parse_method": intent.parse_method,
                "confidence": intent.confidence,
                "entities": str(intent),
                "result": message,
                "success": success,
                "latency_ms": round(latency * 1000, 1),
            },
        )

    def _execute_intent(self, intent: ParsedIntent) -> tuple[bool, str]:
        current_dir = get_current_dir()

        text = intent.text

        if intent.intent == IntentType.CREATE_FILE:
            if not intent.filename:
                return False, "No filename specified"
            target_dir = current_dir
            dest = intent.destination or intent.folder
            if dest:
                resolved = _resolve_destination(dest)
                if resolved:
                    target_dir = resolved
            success, message = create_file(intent.filename, target_dir)
            if success:
                self._remember(target_dir / intent.filename)
            return success, message

        elif intent.intent == IntentType.CREATE_FOLDER:
            if not intent.filename:
                return False, "No folder name specified"
            target_dir = current_dir
            dest = intent.destination or intent.folder
            if dest:
                resolved = _resolve_destination(dest)
                if resolved:
                    target_dir = resolved
            success, message = create_folder(intent.filename, target_dir)
            if success:
                self._remember(target_dir / intent.filename)
            return success, message

        elif intent.intent == IntentType.DELETE:
            if not intent.filename:
                return False, "No filename specified"
            matches = resolve_ambiguous(intent.filename, current_dir)
            if not matches:
                return False, f"Not found: {intent.filename}"
            target = self.confirmer.resolve_ambiguous(matches, "deletion")
            if target is None:
                return False, "Cancelled"
            if self.config.confirm_destructive:
                confirmed = self.confirmer.confirm_destructive(
                    "Move to trash", target.name
                )
                if not confirmed:
                    return False, "Cancelled"
            return delete(target)

        elif intent.intent == IntentType.DUPLICATE:
            if not intent.filename:
                return False, "No filename specified"
            return duplicate_file(intent.filename, current_dir)

        elif intent.intent == IntentType.RENAME:
            if not intent.source or not intent.filename:
                return False, "Need source and target name"
            path = resolve_path(intent.source, current_dir)
            if path is None:
                return False, f"Not found: {intent.source}"
            success, message = rename(path, intent.filename)
            if success:
                self._remember(path.parent / intent.filename)
            return success, message

        elif intent.intent == IntentType.MOVE:
            if not intent.source or not intent.destination:
                return False, "Need source and destination"
            path = resolve_path(intent.source, current_dir)
            if path is None:
                return False, f"Not found: {intent.source}"
            dest_path = _resolve_destination(intent.destination)
            if dest_path is None:
                return False, f"Destination not found: {intent.destination}"
            success, message = move(path, dest_path)
            if success:
                self._remember(dest_path / path.name)
            return success, message

        elif intent.intent == IntentType.COPY:
            if not intent.source or not intent.destination:
                return False, "Need source and destination"
            path = resolve_path(intent.source, current_dir)
            if path is None:
                return False, f"Not found: {intent.source}"
            dest_path = _resolve_destination(intent.destination)
            if dest_path is None:
                return False, f"Destination not found: {intent.destination}"
            success, message = copy_op(path, dest_path)
            if success:
                self._remember(dest_path / path.name)
            return success, message

        
        elif intent.intent == IntentType.OPEN:
    # Detect bulk commands such as "open all files" or "open all PDFs".
            bulk_open = (
            re.search(r"\b(?:all|every)\b", text, re.IGNORECASE)
            and re.search(
            r"\b(?:files?|pdfs?|documents?|docs?|images?|pictures?|"
            r"videos?|python files?)\b",
            text,
            re.IGNORECASE,
        )
    )

            if bulk_open:
         # Choose the file type from the spoken command.
                if re.search(r"\bpdfs?\b", text, re.IGNORECASE):
                    item_type = "pdf"
                elif re.search(r"\b(?:documents?|docs?)\b", text, re.IGNORECASE):
                    item_type = "document"
                elif re.search(r"\b(?:images?|pictures?)\b", text, re.IGNORECASE):
                    item_type = "image"
                elif re.search(r"\bvideos?\b", text, re.IGNORECASE):
                    return False, "Video filtering is not supported yet"
                elif re.search(r"\bpython files?\b", text, re.IGNORECASE):
                    item_type = "python"
                else:
                    item_type = "file"
                success, items = list_items(current_dir, item_type)

                if not success:
                    return False, "Could not list the current directory"

                # Never pass directories to the bulk file-opening operation.
                items = [p for p in items if p.is_file()]

                if not items:
                    return True, f"No matching {item_type} items found"

                # Confirm before launching multiple applications/windows.
                description = f"open all {len(items)} matching {item_type} items"
                if not self.confirmer.confirm_interpretation(description):
                    return False, "Cancelled"

                opened = []
                failed = []

                for path in items:
                    file_success, _ = open_file(path)

                    if file_success:
                        opened.append(path)
                        self._remember(path)
                    else:
                        failed.append(path.name)

                if not opened:
                    return False, f"Could not open any of the {len(items)} matching items"

                message = f"Opened {len(opened)} of {len(items)} matching items"
                if failed:
                    message += f". Failed: {', '.join(failed[:5])}"

                return True, message

            # Original single-file behaviour remains unchanged.
            if not intent.filename:
                return False, "No filename specified"

            path = resolve_path(intent.filename, current_dir)
            if path is None:
                return False, f"Not found: {intent.filename}"

            success, message = open_file(path)
            if success:
                self._remember(path)

            return success, message


        elif intent.intent == IntentType.NAVIGATE:
            dest = intent.destination or ""
            if not dest and intent.filename:
                dest = intent.filename
            if not dest:
                dest = "home"
            success, msg, new_dir = navigate(dest)
            if success:
                set_current_dir(new_dir)
            return success, msg

        elif intent.intent == IntentType.LIST:
            item_type = intent.file_type
            success, items = list_items(current_dir, item_type)
            if success:
                self.ui.show_list_results(items)
                return True, f"Found {len(items)} items"
            return False, "Could not list directory"

        elif intent.intent == IntentType.SEARCH:
            query = intent.filename or text
            results = self.search_engine.search(
                query,
                max_results=self.config.max_search_results,
                current_dir=current_dir,
                recent_paths=list(self._recent_paths),
            )
            self.ui.show_search_results(results, query)
            for r in results[:1]:
                self._remember(r.path)
            return True, f"Found {len(results)} results"

        return False, "Unknown command"

    def run(self) -> None:
        self.initialize()
        self.logger.info("Voice File Assistant started")
        self.confirmer.speak("Voice File Assistant ready")

        try:
            while self._running:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()

    def shutdown(self) -> None:
        self._running = False
        self.recorder.stop_hotkey_listener()
        self.indexer.stop_watching()
        self.logger.info("Voice File Assistant shutdown")


def main() -> None:
    assistant = VoiceFileAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        assistant.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
