from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from voice_assistant.filesystem.operations import known_folder

DEFAULT_CONFIG_PATH = Path.home() / ".voice_assistant" / "config.json"


@dataclass
class Config:
    hotkey: str = "<ctrl>+<lc>"
    indexed_folders: list[str] = field(
        default_factory=lambda: [
            str(known_folder("desktop")),
            str(known_folder("documents")),
            str(known_folder("downloads")),
        ]
    )
    whisper_model: str = "base"
    speech_timeout_seconds: float = 10.0
    confirm_destructive: bool = True
    max_search_results: int = 10
    log_dir: str = field(
        default_factory=lambda: str(Path.home() / ".voice_assistant" / "logs")
    )
    index_dir: str = field(
        default_factory=lambda: str(Path.home() / ".voice_assistant" / "index")
    )

    @classmethod
    def default(cls) -> Config:
        return cls()


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Config = Config.default()
        self._ensure_dirs()
        self.load()

    def _ensure_dirs(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        index_dir = Path(self.config.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Config:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                self.config = Config(**data)
            except (json.JSONDecodeError, TypeError, KeyError):
                self.config = Config.default()
        else:
            # First run: persist the defaults so the file the README documents
            # ("auto-creates default config at ~/.voice_assistant/config.json")
            # actually exists on disk for the user to find and edit.
            self.config = Config.default()
            self.save()
        return self.config

    def save(self) -> None:
        self._ensure_dirs()
        with open(self.config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def get(self) -> Config:
        return self.config
