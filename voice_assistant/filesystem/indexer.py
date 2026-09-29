from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("voice_assistant.filesystem.indexer")

_TEXT_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".csv", ".json", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf"}
_SUPPORTED_DOC = {".txt", ".md", ".py"}


class IndexUpdateHandler(FileSystemEventHandler):
    def __init__(self, indexer: FileIndexer) -> None:
        super().__init__()
        self._indexer = indexer
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._pending = False

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
            if self._indexer._reindexing:
                self._pending = True
                return
            self._debounce_timer = threading.Timer(5.0, self._do_reindex)
            self._debounce_timer.start()

    def _do_reindex(self) -> None:
        self._indexer.reindex()
        with self._lock:
            if self._pending:
                self._pending = False
                self._debounce_timer = threading.Timer(5.0, self._do_reindex)
                self._debounce_timer.start()


class FileIndexer:
    def __init__(self, index_dir: str | Path) -> None:
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._index_dir / "file_index.json"
        self._index: list[dict] = []
        self._watched_folders: list[Path] = []
        self._observer: Optional[Observer] = None
        self._handler: Optional[IndexUpdateHandler] = None
        self._reindexing = False

    def set_watched_folders(self, folders: list[str]) -> None:
        self._watched_folders = [Path(f) for f in folders]

    def build_index(self) -> list[dict]:
        logger.info("Building file index...")
        self._index = []
        seen: set[str] = set()

        for folder in self._watched_folders:
            if not folder.exists():
                logger.warning(f"Watched folder does not exist: {folder}")
                continue
            try:
                for item in folder.rglob("*"):
                    try:
                        if str(item) in seen:
                            continue
                        seen.add(str(item))
                        entry = self._index_entry(item)
                        if entry:
                            self._index.append(entry)
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot scan {folder}: {e}")

        self._save_index()
        logger.info(f"Indexed {len(self._index)} items")
        return self._index

    def reindex(self) -> list[dict]:
        if self._reindexing:
            return self._index
        self._reindexing = True
        try:
            return self.build_index()
        finally:
            self._reindexing = False

    def _index_entry(self, item: Path) -> Optional[dict]:
        try:
            stat = item.stat()
            entry = {
                "filename": item.name,
                "absolute_path": str(item.resolve()),
                "extension": item.suffix.lower() if item.suffix else "",
                "size": stat.st_size,
                "created_date": self._safe_time(stat.st_birthtime),
                "modified_date": self._safe_time(stat.st_mtime),
                "folder": item.parent.name,
                "is_directory": item.is_dir(),
            }
            if item.suffix.lower() in _SUPPORTED_DOC and item.is_file():
                content = self._extract_text_content(item)
                if content:
                    entry["content_preview"] = content[:500]
            return entry
        except (OSError, PermissionError):
            return None

    def _extract_text_content(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _safe_time(self, t: Optional[float]) -> Optional[str]:
        if t and t > 0:
            import datetime
            return datetime.datetime.fromtimestamp(t).isoformat()
        return None

    def _save_index(self) -> None:
        try:
            with open(self._index_file, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load_index(self) -> list[dict]:
        if self._index_file.exists():
            try:
                with open(self._index_file, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                logger.info(f"Loaded {len(self._index)} items from index")
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to load index: {e}")
                self._index = []
        return self._index

    def get_index(self) -> list[dict]:
        return self._index if self._index else self.load_index()

    def start_watching(self) -> None:
        if self._observer:
            return
        self._handler = IndexUpdateHandler(self)
        self._observer = Observer()
        for folder in self._watched_folders:
            if folder.exists():
                self._observer.schedule(self._handler, str(folder), recursive=True)
                logger.info(f"Watching folder: {folder}")
        self._observer.start()

    def stop_watching(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("File watcher stopped")
