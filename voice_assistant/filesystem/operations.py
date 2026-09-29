from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import send2trash

logger = logging.getLogger("voice_assistant.filesystem.operations")

_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
}


def _known_folder_win(name: str) -> Optional[str]:
    """Resolve a Windows known folder via SHGetKnownFolderPath."""
    if sys.platform != "win32":
        return None
    folder_id = _FOLDER_IDS.get(name)
    if not folder_id:
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        shell32 = ctypes.windll.shell32
        get_path = shell32.SHGetKnownFolderPath
        get_path.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        get_path.restype = wintypes.HRESULT
        raw = ctypes.c_wchar_p()
        hr = get_path(uuid.UUID(folder_id).bytes_le, 0, None, ctypes.byref(raw))
        if hr == 0 and raw.value:
            path = raw.value
            ctypes.windll.ole32.CoTaskMemFree(raw)
            return path
    except Exception:
        pass
    return None


def known_folder(name: str) -> Path:
    """Resolve a known folder (desktop/documents/downloads/projects/home).

    Uses the Windows Known Folder API so redirected folders (e.g. an
    OneDrive desktop) resolve to the folder actually shown in Explorer.
    Falls back to the conventional location under the home directory.
    """
    path = _known_folder_win(name)
    if path:
        return Path(path)
    home = Path.home()
    key = name.strip().lower().rstrip("s")
    fallbacks = {
        "desktop": home / "Desktop",
        "document": home / "Documents",
        "download": home / "Downloads",
        "project": home / "Projects",
    }
    return fallbacks.get(key, home)

_NAV_HISTORY: list[Path] = []
_CURRENT_DIR: Path = Path.home()


def get_current_dir() -> Path:
    return _CURRENT_DIR


def set_current_dir(path: Path) -> None:
    global _CURRENT_DIR
    _CURRENT_DIR = path


def create_file(name: str, folder: Optional[Path] = None) -> tuple[bool, str]:
    target = (folder or _CURRENT_DIR) / name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return False, f"File already exists: {target.name}"
        target.touch()
        logger.info(f"Created file: {target}")
        return True, f"Created file: {target.name}"
    except OSError as e:
        return False, f"Cannot create file: {e}"


def create_folder(name: str, folder: Optional[Path] = None) -> tuple[bool, str]:
    target = (folder or _CURRENT_DIR) / name
    try:
        target.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder: {target}")
        return True, f"Created folder: {target.name}"
    except OSError as e:
        return False, f"Cannot create folder: {e}"


def delete(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Not found: {path.name}"
    try:
        send2trash.send2trash(str(path))
        logger.info(f"Moved to trash: {path}")
        return True, f"Moved to trash: {path.name}"
    except Exception as e:
        return False, f"Cannot trash {path.name}: {e}"


def rename(path: Path, new_name: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Not found: {path.name}"
    try:
        new_path = path.parent / new_name
        if new_path.exists():
            return False, f"Target already exists: {new_name}"
        path.rename(new_path)
        logger.info(f"Renamed {path.name} -> {new_name}")
        return True, f"Renamed {path.name} to {new_name}"
    except OSError as e:
        return False, f"Cannot rename: {e}"


def move(path: Path, destination: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Not found: {path.name}"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        dest_path = destination / path.name
        shutil.move(str(path), str(dest_path))
        logger.info(f"Moved {path.name} -> {destination}")
        return True, f"Moved {path.name} to {destination.name}"
    except (shutil.Error, OSError) as e:
        return False, f"Cannot move: {e}"


def duplicate_file(name: str, search_in: Optional[Path] = None) -> tuple[bool, str]:
    path = resolve_path(name, search_in)
    if path is None:
        return False, f"Not found: {name}"
    if not path.is_file():
        return False, f"Not a file: {path.name}"
    try:
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            copy_name = f"{stem} copy{counter}{suffix}" if counter > 1 else f"{stem} copy{suffix}"
            dest = parent / copy_name
            if not dest.exists():
                break
            counter += 1
        shutil.copy2(str(path), str(dest))
        logger.info(f"Duplicated {path.name} -> {dest.name}")
        return True, f"Created copy: {dest.name}"
    except (shutil.Error, OSError) as e:
        return False, f"Cannot duplicate: {e}"


def copy_op(path: Path, destination: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Not found: {path.name}"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        dest_path = destination / path.name
        if path.is_dir():
            shutil.copytree(str(path), str(dest_path), dirs_exist_ok=True)
        else:
            shutil.copy2(str(path), str(dest_path))
        logger.info(f"Copied {path.name} -> {destination}")
        return True, f"Copied {path.name} to {destination.name}"
    except (shutil.Error, OSError) as e:
        return False, f"Cannot copy: {e}"


def open_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Not found: {path.name}"
    try:
        if sys.platform == "win32":
            # os.startfile is the correct cross-platform-safe way to invoke
            # the OS default handler on Windows — subprocess.Popen(["open", ...])
            # is a macOS-only command and doesn't exist on Windows at all.
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        logger.info(f"Opened: {path}")
        return True, f"Opened: {path.name}"
    except OSError as e:
        return False, f"Cannot open: {e}"


def navigate(target: str | Path) -> tuple[bool, str, Path]:
    global _CURRENT_DIR
    target_str = str(target).lower().strip()
    known = {
        "home": Path.home(),
        "desktop": known_folder("desktop"),
        "documents": known_folder("documents"),
        "downloads": known_folder("downloads"),
        "projects": known_folder("projects"),
    }
    if target_str in known:
        new_dir = known[target_str]
    elif target_str == "up":
        new_dir = _CURRENT_DIR.parent
    elif target_str == "back":
        if len(_NAV_HISTORY) > 1:
            _NAV_HISTORY.pop()
            new_dir = _NAV_HISTORY[-1]
        else:
            return False, "No navigation history", _CURRENT_DIR
    else:
        candidate = Path(target_str)
        if candidate.is_absolute():
            new_dir = candidate
        else:
            new_dir = _CURRENT_DIR / target_str
    if not new_dir.exists():
        return False, f"Directory not found: {new_dir.name}", _CURRENT_DIR
    if not new_dir.is_dir():
        return False, f"Not a directory: {new_dir.name}", _CURRENT_DIR
    _NAV_HISTORY.append(_CURRENT_DIR)
    _CURRENT_DIR = new_dir
    logger.info(f"Navigated to: {_CURRENT_DIR}")
    return True, f"Now in: {_CURRENT_DIR.name}", _CURRENT_DIR


def list_items(
    path: Optional[Path] = None,
    item_type: Optional[str] = None,
) -> tuple[bool, list[Path]]:
    target = path or _CURRENT_DIR
    if not target.exists() or not target.is_dir():
        return False, []
    try:
        items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if item_type:
            item_type = item_type.lower().strip("s")
            type_map = {
                "file": lambda p: p.is_file(),
                "folder": lambda p: p.is_dir(),
                "directory": lambda p: p.is_dir(),
                "pdf": lambda p: p.is_file() and p.suffix.lower() == ".pdf",
                "text": lambda p: p.is_file() and p.suffix.lower() == ".txt",
                "python": lambda p: p.is_file() and p.suffix.lower() == ".py",
                "image": lambda p: p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp"},
                "document": lambda p: p.is_file() and p.suffix.lower() in {".doc", ".docx"},
            }
            filter_fn = type_map.get(item_type)
            if filter_fn:
                items = [p for p in items if filter_fn(p)]
            else:
                ext = f".{item_type}" if not item_type.startswith(".") else item_type
                items = [p for p in items if p.suffix.lower() == ext]
        return True, items
    except PermissionError:
        return False, []


def resolve_path(name: str, search_in: Optional[Path] = None) -> Optional[Path]:
    locations = [search_in or _CURRENT_DIR]
    if search_in is None:
        for sub in ["desktop", "documents", "downloads"]:
            locations.append(known_folder(sub))
    for loc in locations:
        if not loc.exists():
            continue
        candidate = loc / name
        if candidate.exists():
            return candidate.resolve()
        for item in loc.iterdir():
            if item.name.lower() == name.lower():
                return item.resolve()
    for loc in locations:
        if not loc.exists():
            continue
        for item in loc.iterdir():
            stem_sim = item.stem.lower() == Path(name).stem.lower()
            if stem_sim:
                return item.resolve()
    return None


def resolve_ambiguous(
    name: str, search_in: Optional[Path] = None
) -> list[Path]:
    matches: list[Path] = []
    locations = [search_in or _CURRENT_DIR]
    if search_in is None:
        for sub in ["desktop", "documents", "downloads"]:
            locations.append(known_folder(sub))
    seen: set[str] = set()
    for loc in locations:
        if not loc.exists():
            continue
        for item in loc.iterdir():
            if str(item) in seen:
                continue
            seen.add(str(item))
            name_lower = name.lower()
            item_stem = item.stem.lower()
            item_name = item.name.lower()
            if name_lower in item_name or item_stem in name_lower:
                matches.append(item.resolve())
            elif item_stem.startswith(name_lower) or name_lower.startswith(item_stem):
                matches.append(item.resolve())
    return matches