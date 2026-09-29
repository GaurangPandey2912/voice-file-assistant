from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from voice_assistant.intent.parser import ParsedIntent, IntentType


class UIFeedback:
    def __init__(self, hotkey_label: str = "Left Ctrl") -> None:
        self._term_width = shutil.get_terminal_size().columns
        self._hotkey_label = hotkey_label

    def _separator(self, char: str = "\u2500") -> str:
        return char * self._term_width

    def header(self) -> None:
        print()
        print(f"  Voice File Assistant  |  {self._hotkey_label}: push-to-talk  |  Ctrl+C: quit")
        sep = self._separator()
        print(f"  {sep}")

    def show_current_dir(self, path: Path) -> None:
        print(f"  \u2514 Current: {path}")

    def show_speech(self, text: str) -> None:
        print(f"\n  \u2591 Recognized: \"{text}\"")

    def show_intent(self, intent: ParsedIntent) -> None:
        intent_name = intent.intent.name.replace("_", " ").title()
        method = getattr(intent, "parse_method", "regex")
        print(
            f"  \u2591 Intent: {intent_name} (confidence: {intent.confidence:.0%}, "
            f"method: {method})"
        )
        entities = []
        if intent.filename:
            entities.append(f"file={intent.filename}")
        if intent.source:
            entities.append(f"source={intent.source}")
        if intent.destination:
            entities.append(f"destination={intent.destination}")
        if intent.file_type:
            entities.append(f"type={intent.file_type}")
        if intent.folder:
            entities.append(f"folder={intent.folder}")
        if intent.date_expr:
            entities.append(f"date={intent.date_expr}")
        if intent.keywords:
            entities.append(f"keywords={intent.keywords}")
        if entities:
            print(f"  \u2591 Entities: {', '.join(entities)}")

    def show_result(self, success: bool, message: str) -> None:
        icon = "\u2713" if success else "\u2717"
        print(f"  {icon} {message}")

    def show_error(self, message: str) -> None:
        print(f"  \u26a0 Error: {message}")

    def show_search_results(self, results: list, query: str) -> None:
        if not results:
            print(f"  \u2139 No results for: \"{query}\"")
            return
        print(f"  \u2139 Search results for: \"{query}\"")
        for i, r in enumerate(results, 1):
            extra = ""
            if hasattr(r, "match_type") and r.match_type:
                extra = f" [{r.match_type}]"
            print(f"    {i}. {r.name}{extra}")
            print(f"       \u2514 {r.path.parent}")
        print(f"  Found {len(results)} result(s)")

    def show_list_results(self, items: list[Path]) -> None:
        if not items:
            print("  (empty)")
            return
        folders = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file()]
        if folders:
            print(f"  Folders ({len(folders)}):")
            for f in folders:
                print(f"    \u2514 {f.name}/")
        if files:
            print(f"  Files ({len(files)}):")
            for f in files:
                size = f.stat().st_size if f.exists() else 0
                size_str = self._format_size(size)
                print(f"    \u2514 {f.name}  ({size_str})")

    def _format_size(self, size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
