# Voice File Assistant

An offline-first, privacy-preserving voice-controlled file manager designed for users with motor impairments. Control your file system entirely through voice — create, delete (to trash), rename, move, copy, open files, navigate directories, and search files using natural language.

## What's new in v0.2

- **spaCy fallback parsing** — regex still handles common phrasings first (fast, deterministic), but anything it misses now falls through to a spaCy dependency-parse layer that maps the sentence's main verb (via lemmatization) to an intent and pulls the filename/destination from noun chunks. This is what the architecture diagram's "Intent & Entity Parser: spaCy + Regex" box actually refers to.
- **Context-aware hybrid search** — `search/engine.py` now combines four signals: filename relevance (exact/token/fuzzy), semantic similarity (sentence-transformers embeddings over filename + content preview), a boost for files in the user's **current directory**, and a decaying boost for **recently interacted-with files** this session.
- **Confidence-based decision layer** — a confident regex match executes immediately; a spaCy-fallback or keyword-fallback match (anything below 75% confidence) is read back to the user for a yes/no confirmation before it runs.
- **Config auto-persistence fix** — `ConfigManager` now actually writes the default config to disk on first run, matching what the docs below always claimed.
- **Test suite** — `tests/` covers the intent parser, search ranking, filesystem operations, and config manager (30 tests, no audio hardware needed).

## Features

- **Push-to-talk**: Hold Right Ctrl, speak, release — intent is parsed and executed
- **File operations**: Create, delete (trash-safe), rename, move, copy, open
- **Navigation**: Go to Desktop, Downloads, Documents, or any directory with history support
- **List**: List files, folders, PDFs, Python files, etc.
- **Intelligent search**: Find files by name, date, type, folder, and fuzzy matching — "find yesterday's networking notes"
- **Safety first**: Always confirm deletions, resolve ambiguous filenames
- **Offline & private**: Everything runs locally — no data leaves your machine
- **File index**: Pre-indexed Desktop, Documents, Downloads with automatic watchdog updates
- **Audio feedback**: Speaks errors and confirmations via TTS

## Requirements

- Python 3.10+
- macOS, Linux, or Windows

## Installation

```bash
# Clone or extract the project
cd voice_assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python3 -m spacy download en_core_web_sm
```

## Quick Start

```bash
# From the project root
python3 -m voice_assistant.main
```

The assistant will:
1. Load its configuration (auto-creates default config at `~/.voice_assistant/config.json`)
2. Build a file index of Desktop, Documents, and Downloads
3. Start watching those folders for changes
4. Display the UI and begin listening for the hotkey

**Press and hold Left Ctrl**, speak a command, then release. The assistant processes and executes your command.

Press `Ctrl+C` to quit.

## Supported Commands

### File Creation
```
"create file project notes"
"create folder Networking"
"make folder Semester 6"
```

### Delete (moves to trash — never permanent)
```
"delete report.pdf"
"delete networking folder"
```
Always prompts for confirmation.

### Rename
```
"rename report to final report"
"rename notes.txt to networking notes.txt"
```

### Move
```
"move report.pdf to Documents"
"move all PDFs to Downloads"
```

### Copy
```
"copy invoice.pdf to Desktop"
```

### Open (OS default application)
```
"open networking notes"
"open GST bill"
```

### Navigation
```
"go to Downloads"
"go home"
"go up"
"go back"
```

### List
```
"list files"
"list folders"
"list PDFs"
```

### Intelligent Search
```
"find yesterday's networking notes"
"find final GST bill"
"find all invoices from Downloads"
"find Python files from last week"
"find notes about routing"
"find resume"
"find assignment submitted yesterday"
```

## Configuration

Auto-created at `~/.voice_assistant/config.json`:

```json
{
  "hotkey": "<ctrl>+<rc>",
  "indexed_folders": [
    "/Users/username/Desktop",
    "/Users/username/Documents",
    "/Users/username/Downloads"
  ],
  "whisper_model": "base",
  "speech_timeout_seconds": 10.0,
  "confirm_destructive": true,
  "max_search_results": 10
}
```

Edit this file while the assistant is stopped to change settings.

## Architecture

```
voice_assistant/
├── __init__.py
├── main.py                  # Orchestrator — pipeline & lifecycle
├── audio/
│   ├── __init__.py
│   └── recorder.py          # Hotkey listener + sounddevice recording
├── speech/
│   ├── __init__.py
│   └── transcriber.py       # Whisper local transcription
├── intent/
│   ├── __init__.py
│   └── parser.py            # spaCy + regex intent/entity extraction
├── search/
│   ├── __init__.py
│   └── engine.py            # Multi-technique file search
├── filesystem/
│   ├── __init__.py
│   ├── indexer.py           # File index with watchdog live updates
│   └── operations.py        # All file operations (create, delete, etc.)
├── confirmation/
│   ├── __init__.py
│   └── handler.py           # Safety checks, TTS, conflict resolution
├── ui/
│   ├── __init__.py
│   └── feedback.py          # Terminal UI display
├── config/
│   ├── __init__.py
│   ├── config_manager.py    # JSON config load/save
│   └── example_config.json
└── logs/
    ├── __init__.py
    └── logger.py            # Structured logging to file + console
```

### Pipeline

```
User presses hotkey → Recording starts → User releases hotkey
  → Whisper transcribes → Intent parser → Safety layer
  → Execute operation → Visual feedback → TTS if needed
```

### Search Pipeline

```
Speech → Extract query → Extract dates (dateparser)
  → Extract keywords → Identify file type → Identify folder
  → Metadata filtering → Filename similarity (fuzzy)
  → (future: semantic embeddings) → Rank → Return top results
```

## Module Documentation

### `audio/recorder.py`
- `AudioRecorder` — captures audio via `sounddevice` when Right Ctrl is held
- Runs `pynput` keyboard listener in background thread
- Returns `numpy.ndarray` of audio samples at 16kHz

### `speech/transcriber.py`
- `SpeechTranscriber` — wraps OpenAI Whisper for local transcription
- Caches loaded model; defaults to `base` model for speed/accuracy balance
- Returns transcribed text or `None` on failure

### `intent/parser.py`
- `IntentParser` — three-tier intent classification:
  1. **Regex patterns** (fast path, confidence 1.0) for common phrasings.
  2. **spaCy fallback** (confidence 0.65): lemmatizes the main verb, maps it to an intent via `_VERB_INTENT_MAP`, and extracts filename/destination from noun chunks and their governing prepositions. Handles phrasing regex doesn't cover (e.g. "erase" instead of "delete").
  3. **Keyword fallback** (confidence 0.4): last resort, treats the utterance as a search query.
- Identifies: CREATE_FILE, CREATE_FOLDER, DELETE, RENAME, MOVE, COPY, OPEN, NAVIGATE, LIST, SEARCH
- Extracts: filename, source, destination, file_type, folder, date expressions
- Uses `dateparser` for relative date understanding, with spaCy NER (`DATE` entities) as a second pass when `dateparser` doesn't find a match
- `ParsedIntent.is_low_confidence` (confidence < 0.75) drives the confirmation step in `main.py`

### `search/engine.py` + `search/semantic.py`
- `SearchEngine` — hybrid, context-aware ranking that combines:
  - Exact filename matching (highest score)
  - Fuzzy string matching via `difflib.SequenceMatcher`
  - Token/partial word matching, file type filtering, folder/location matching
  - **Semantic similarity** — cosine similarity between the query and each file's filename + content-preview embedding, via `SemanticEmbedder` (`sentence-transformers`, `all-MiniLM-L6-v2`). Lazily loaded and cached per-file; degrades gracefully to keyword-only scoring if the package/model isn't available.
  - **Current-directory context** — bonus for files already in the folder the user is standing in
  - **Recent-interaction context** — decaying bonus for files created/opened/moved this session (tracked in `main.py`'s `_recent_paths`)
- Returns ranked `SearchResult` list with score and match type (e.g. `"exact+semantic+current_dir"`)

### `filesystem/indexer.py`
- `FileIndexer` — builds and maintains a JSON index of watched folders
- Stores: filename, absolute path, extension, size, created/modified dates, folder
- Extracts text content preview from `.txt`, `.md`, `.py` files
- `watchdog` observer for live index updates (2-second debounce)
- Saves index to `~/.voice_assistant/index/file_index.json`

### `filesystem/operations.py`
- File operations with safety validation:
  - `create_file`, `create_folder` — with parent dir creation
  - `delete` — uses `send2trash` (never permanent)
  - `rename`, `move`, `copy` — with path validation
  - `open_file` — uses OS default via `subprocess.Popen(["open", ...])`
  - `navigate` — supports named locations plus history stack
  - `list_items` — filtered listing by type
  - `resolve_path`, `resolve_ambiguous` — filename disambiguation

### `confirmation/handler.py`
- `ConfirmationHandler` — safety layer for destructive operations
- `confirm_destructive()` — prompts yes/no before trash operations
- `resolve_ambiguous()` — when multiple files match, lists them and asks for selection
- `validate_path()` — ensures paths exist before operations
- TTS integration via `pyttsx3` for speaking confirmations and errors

### `ui/feedback.py`
- `UIFeedback` — terminal UI with Unicode box-drawing characters
- Displays: current directory, recognized speech, detected intent, entities, results, search results, file listings with sizes

### `config/config_manager.py`
- `ConfigManager` — loads/saves JSON config from `~/.voice_assistant/config.json`
- `Config` dataclass with defaults for all settings
- Auto-creates config and support directories on first run

### `logs/logger.py`
- Structured logging with timestamp, level, message, and extra fields
- Dual output: console (stdout) + rotating file in `~/.voice_assistant/logs/`
- Logs every pipeline execution: speech, intent, entities, result, latency

## Safety Features

- **Trash-only deletion**: Files are never permanently deleted — always moved to system trash via `send2trash`
- **Destructive confirmation**: Every delete operation requires explicit yes/no confirmation
- **Ambiguity resolution**: If multiple files match a command, the system lists them and asks for a number or exact name — never guesses
- **Path validation**: All move/copy operations verify source and destination exist before executing
- **Error feedback**: Errors are both displayed and spoken via TTS

## Future Compatibility

The architecture is designed for easy addition of:
- Semantic embedding search (`sentence-transformers`)
- Voice confirmations (yes/no via speech)
- Wake-word activation
- GUI/system tray
- Undo operations
- LLM-powered intent understanding
- OCR for scanned PDFs
- Continuous conversation mode

## Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q
```

The suite (30 tests) covers the intent parser (regex + spaCy fallback paths), the hybrid search engine's ranking and context boosts, filesystem operations, and config persistence — none of it needs a microphone, Whisper, or a downloaded semantic model (`SearchEngine` is instantiated with `use_semantic=False` in tests to keep them fast and offline-safe).

## Logs

Session logs are stored at `~/.voice_assistant/logs/session_YYYYMMDD_HHMMSS.log` with structured entries for every command:

```
2026-06-26T10:15:45.678 [INFO] Pipeline completed | speech="find yesterday's networking notes" | intent=SEARCH | result="Found 3 results" | success=True | latency_ms=2340.5
```

## License

MIT
