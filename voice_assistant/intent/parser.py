from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import dateparser
import spacy
from spacy.language import Language

logger = logging.getLogger("voice_assistant.intent")


class IntentType(Enum):
    CREATE_FILE = auto()
    CREATE_FOLDER = auto()
    DELETE = auto()
    DUPLICATE = auto()
    RENAME = auto()
    MOVE = auto()
    COPY = auto()
    OPEN = auto()
    NAVIGATE = auto()
    LIST = auto()
    SEARCH = auto()
    UNKNOWN = auto()


@dataclass
class ParsedIntent:
    intent: IntentType = IntentType.UNKNOWN
    text: str = ""
    filename: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    file_type: Optional[str] = None
    folder: Optional[str] = None
    date_expr: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parse_method: str = "unmatched"  # "regex" | "spacy_fallback" | "keyword_fallback"

    @property
    def is_destructive(self) -> bool:
        return self.intent in (IntentType.DELETE,)

    @property
    def needs_confirmation(self) -> bool:
        return self.intent in (IntentType.DELETE, IntentType.MOVE, IntentType.COPY)

    @property
    def is_low_confidence(self) -> bool:
        """True when the parse was a best-effort guess rather than a confident
        pattern match, signalling that the caller should confirm before acting."""
        return self.confidence < 0.75


_LOCATION_STRIP = re.compile(
    r"\s+(?:on|in|to|into)\s+(?:the\s+)?(.+?)\s*$", re.IGNORECASE
)

# Lemma -> intent map used by the spaCy fallback parser. Regex patterns cover the
# common phrasings; this covers everything else (different verb tense, synonyms,
# reordered phrasing) via lemmatization + dependency parsing instead of needing
# every inflection hard-coded as its own regex alternative.
_VERB_INTENT_MAP: dict[str, IntentType] = {
    "create": IntentType.CREATE_FILE,
    "make": IntentType.CREATE_FILE,
    "add": IntentType.CREATE_FILE,
    "delete": IntentType.DELETE,
    "remove": IntentType.DELETE,
    "erase": IntentType.DELETE,
    "trash": IntentType.DELETE,
    "discard": IntentType.DELETE,
    "duplicate": IntentType.DUPLICATE,
    "clone": IntentType.DUPLICATE,
    "copy": IntentType.COPY,
    "rename": IntentType.RENAME,
    "relabel": IntentType.RENAME,
    "move": IntentType.MOVE,
    "shift": IntentType.MOVE,
    "relocate": IntentType.MOVE,
    "transfer": IntentType.MOVE,
    "open": IntentType.OPEN,
    "launch": IntentType.OPEN,
    "start": IntentType.OPEN,
    "go": IntentType.NAVIGATE,
    "navigate": IntentType.NAVIGATE,
    "list": IntentType.LIST,
    "show": IntentType.LIST,
    "display": IntentType.LIST,
    "find": IntentType.SEARCH,
    "search": IntentType.SEARCH,
    "locate": IntentType.SEARCH,
    "look": IntentType.SEARCH,
    "get": IntentType.SEARCH,
}

_FOLDER_NOUNS = {"folder", "directory"}
_DEST_PREPOSITIONS = {"to", "into", "in", "on"}

# Whisper (and ASR generally) tends to punctuate transcripts like written
# sentences — "create a folder called ABC on desktop." — which would
# otherwise make "desktop." fail to match the known folder "desktop".
_TRAILING_PUNCT = re.compile(r"[.,!?;:]+\s*$")

# Filler prefixes that show up in natural phrasing ("move the file X...",
# "rename the folder Y...") but aren't part of the actual name.
_FILLER_PREFIXES = [
    "the file ", "a file ", "an file ", "file ",
    "the folder ", "a folder ", "an folder ", "folder ",
    "the ", "a ", "an ",
]

# Generic words that precede "file(s)" in ordinary sentences but are not
# themselves a file type/extension — guards a too-loose fallback pattern.
_GENERIC_FILE_WORDS = {
    "a", "an", "the", "this", "that", "new", "old", "my", "your", "some",
    "any", "called", "named", "single", "one",
    # Command verbs also shouldn't be mistaken for a file type/extension.
    "create", "make", "add", "delete", "remove", "erase", "trash", "discard",
    "duplicate", "clone", "copy", "rename", "relabel", "move", "shift",
    "relocate", "transfer", "open", "launch", "start", "go", "navigate",
    "list", "show", "display", "find", "search", "locate", "look", "get",
}


def _strip_trailing_punctuation(text: str) -> str:
    return _TRAILING_PUNCT.sub("", text).strip()


def _strip_filler_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in _FILLER_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _strip_from_clause(text: str) -> str:
    """Drops a trailing '... from <location>' clause from a MOVE/COPY/RENAME
    source phrase, e.g. 'xyz.py from desktop' -> 'xyz.py'. The location is
    redundant: resolve_path() already searches the common known folders."""
    m = re.search(r"\s+from\s+.+$", text, re.IGNORECASE)
    if m:
        return text[: m.start()].strip()
    return text


def _clean_extracted(text: Optional[str], is_source: bool = False) -> Optional[str]:
    if not text:
        return text
    cleaned = _strip_trailing_punctuation(text)
    cleaned = _strip_filler_prefix(cleaned)
    cleaned = _strip_trailing_punctuation(cleaned)
    if is_source:
        cleaned = _strip_from_clause(cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def _load_spacy_model() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' is not installed. Run "
            "'python -m spacy download en_core_web_sm' for full accuracy. "
            "Falling back to a blank English pipeline (tokenization only, no NER/lemmas)."
        )
        nlp = spacy.blank("en")
        return nlp


class IntentParser:
    def __init__(self) -> None:
        self._load_patterns()
        self._nlp: Optional[Language] = None

    @property
    def nlp(self) -> Language:
        """Lazily load the spaCy pipeline — it's only needed once regex misses,
        so most commands never pay the load cost."""
        if self._nlp is None:
            self._nlp = _load_spacy_model()
        return self._nlp

    def _strip_trailing_location(self, text: str) -> tuple[str, Optional[str]]:
        m = _LOCATION_STRIP.search(text)
        if m:
            dest = m.group(1).strip().lower()
            return text[:m.start()].strip(), dest
        return text, None

    def _load_patterns(self) -> None:
        self._patterns: list[tuple[re.Pattern, IntentType, list[str]]] = [
            (
                re.compile(
                    r"(?:create|make)\s+a\s+copy\s+of\s+(.+)", re.IGNORECASE
                ),
                IntentType.DUPLICATE,
                ["filename"],
            ),
            (
                re.compile(r"(?:duplicate)\s+(.+)", re.IGNORECASE),
                IntentType.DUPLICATE,
                ["filename"],
            ),
            (
                re.compile(
                    r"(?:create|make|new)\s+(?:(?:a|new)\s+)*(?:folder|directory)\s+(?:called\s+|named\s+)?(.+?)(?:\s+(?:on|in|to|into)\s+(?:the\s+)?(.+))?$",
                    re.IGNORECASE,
                ),
                IntentType.CREATE_FOLDER,
                ["filename", "destination"],
            ),
            (
                re.compile(
                    r"(?:create|make|new)\s+(?:(?:a|new)\s+)*(?:file\s+)?(?:called\s+|named\s+)?(.+?)(?:\s+(?:on|in|to|into)\s+(?:the\s+)?(.+))?$",
                    re.IGNORECASE,
                ),
                IntentType.CREATE_FILE,
                ["filename", "destination"],
            ),
            (
                re.compile(r"(?:delete|remove|trash)\s+(.+)", re.IGNORECASE),
                IntentType.DELETE,
                ["filename"],
            ),
            (
                re.compile(
                    r"(?:rename)\s+(.+?)\s+(?:to)\s+(.+)", re.IGNORECASE
                ),
                IntentType.RENAME,
                ["source", "filename"],
            ),
            (
                re.compile(
                    r"(?:move)\s+(.+?)\s+(?:to|into|in)\s+(?:(?:the\s+)?(.+))",
                    re.IGNORECASE,
                ),
                IntentType.MOVE,
                ["source", "destination"],
            ),
            (
                re.compile(
                    r"(?:copy)\s+(.+?)\s+(?:to|into|in)\s+(?:(?:the\s+)?(.+))",
                    re.IGNORECASE,
                ),
                IntentType.COPY,
                ["source", "destination"],
            ),
            (
                re.compile(r"(?:open|launch)\s+(.+)", re.IGNORECASE),
                IntentType.OPEN,
                ["filename"],
            ),
            (
                re.compile(
                    r"(?:change\s+(?:directory|folder)\s+to|go\s+to|navigate\s+to|change\s+to)\s+(.+)",
                    re.IGNORECASE,
                ),
                IntentType.NAVIGATE,
                ["destination"],
            ),
            (
                re.compile(
                    r"(?:list|show|display)\s+(?:all\s+)?(?:(.+?)\s+)?(?:files|folders|items|documents|pdfs|pictures|images|videos)",
                    re.IGNORECASE,
                ),
                IntentType.LIST,
                ["file_type"],
            ),
            (
                re.compile(
                    r"(?:list|show)\s+(?:all\s+)?(?:files|folders|items)",
                    re.IGNORECASE,
                ),
                IntentType.LIST,
                [],
            ),
            (
                re.compile(
                    r"(?:find|search|locate|look\s+for)\s+(.+)",
                    re.IGNORECASE,
                ),
                IntentType.SEARCH,
                ["filename"],
            ),
        ]

    def _extract_file_type(self, text: str) -> Optional[str]:
        type_patterns = {
            "pdf": r"\bpdfs?\b",
            "python": r"\bpython\b",
            "text": r"\btexts?\b",
            "image": r"\b(?:images?|pictures?|jpe?gs?|pngs?|gifs?)\b",
            "document": r"\b(?:documents?|docs?|docx)\b",
            "folder": r"\b(?:folders?|directory|directories)\b",
            "markdown": r"\b(?:markdown|md)\b",
        }
        for ftype, pattern in type_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return ftype
        ext_match = re.search(r"\b(\w+)\s+files?\b", text, re.IGNORECASE)
        if ext_match:
            candidate = ext_match.group(1).lower()
            if candidate not in _GENERIC_FILE_WORDS:
                return candidate
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        settings = {"PREFER_DATES_FROM": "past"}
        try:
            date = dateparser.parse(text, settings=settings)
        except Exception:
            date = None
        if date:
            return date.isoformat()

        # Fall back to spaCy's NER for DATE entities (e.g. "last Tuesday",
        # "the 3rd of June") that dateparser's own scan didn't catch whole.
        try:
            doc = self.nlp(text)
            for ent in getattr(doc, "ents", []):
                if ent.label_ == "DATE":
                    parsed = dateparser.parse(ent.text, settings=settings)
                    if parsed:
                        return parsed.isoformat()
                    return ent.text
        except Exception as e:
            logger.debug(f"spaCy NER date extraction skipped: {e}")

        date_keywords = [
            r"\byesterday\b",
            r"\btoday\b",
            r"\blast\s+\w+\b",
            r"\bthis\s+\w+\b",
            r"\bprevious\s+\w+\b",
            r"\bago\b",
            r"\b\d{4}\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
        ]
        for pattern in date_keywords:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_folder(self, text: str, intent: IntentType) -> Optional[str]:
        known_folders = {
            "desktop": "Desktop",
            "documents": "Documents",
            "downloads": "Downloads",
            "projects": "Projects",
            "home": str(Path.home()),
        }
        if intent in (IntentType.MOVE, IntentType.COPY, IntentType.CREATE_FILE, IntentType.CREATE_FOLDER):
            to_match = re.search(
                r"(?:to|into|in|on)\s+(?:the\s+)?(.+)", text, re.IGNORECASE
            )
            if to_match:
                dest = to_match.group(1).strip().lower()
                return known_folders.get(dest, dest)
        if intent == IntentType.NAVIGATE:
            nav_match = re.search(
                r"(?:go|navigate|change)\s+(?:to\s+)?(.+)", text, re.IGNORECASE
            )
            if nav_match:
                dest = nav_match.group(1).strip().lower()
                return known_folders.get(dest, dest)
        return None

    def _build_keywords(self, parsed: ParsedIntent) -> list[str]:
        keywords: list[str] = []
        if parsed.filename:
            keywords.extend(parsed.filename.split())
        if parsed.source:
            keywords.extend(parsed.source.split())
        if parsed.destination and parsed.destination not in (
            "desktop", "documents", "downloads", "projects", "home",
        ):
            keywords.append(parsed.destination)
        return keywords

    def _regex_parse(self, text_clean: str, original_text: str) -> Optional[ParsedIntent]:
        for pattern, intent_type, groups in self._patterns:
            match = pattern.search(text_clean)
            if not match:
                continue

            parsed = ParsedIntent(intent=intent_type, text=original_text)
            group_names = {name: match.group(i + 1) for i, name in enumerate(groups)}
            parsed.filename = _clean_extracted(group_names.get("filename"))
            parsed.source = _clean_extracted(group_names.get("source"), is_source=True)
            parsed.destination = _clean_extracted(group_names.get("destination"))

            if intent_type == IntentType.CREATE_FILE and parsed.filename:
                first = parsed.filename.split(maxsplit=1)
                if first and first[0].lower() in ("folder", "directory"):
                    parsed.intent = IntentType.CREATE_FOLDER
                    parsed.filename = first[1] if len(first) > 1 else first[0]

            if parsed.filename and parsed.intent in (
                IntentType.CREATE_FILE, IntentType.CREATE_FOLDER,
            ):
                stripped, loc = self._strip_trailing_location(parsed.filename)
                if loc:
                    parsed.filename = stripped
                    if not parsed.destination:
                        parsed.destination = loc

            parsed.file_type = self._extract_file_type(text_clean)
            parsed.folder = self._extract_folder(text_clean, parsed.intent)
            parsed.date_expr = self._extract_date(text_clean)
            parsed.keywords = self._build_keywords(parsed)
            parsed.confidence = 1.0
            parsed.parse_method = "regex"
            logger.debug(
                f"Regex parse: intent={parsed.intent.name}, filename={parsed.filename}, "
                f"source={parsed.source}, dest={parsed.destination}"
            )
            return parsed
        return None

    def _spacy_fallback_parse(self, text_clean: str, original_text: str) -> Optional[ParsedIntent]:
        """Dependency-parse based fallback for phrasings the regex layer misses.
        Finds the main verb, maps its lemma to an intent, then walks the verb's
        direct object / prepositional-object noun chunks for filename & destination.
        """
        try:
            doc = self.nlp(text_clean)
        except Exception as e:
            logger.debug(f"spaCy fallback parse unavailable: {e}")
            return None

        if not doc.has_annotation("DEP"):
            # No trained pipeline loaded (en_core_web_sm missing) — blank
            # pipelines have no tagger/parser/lemmatizer, so there's no
            # dependency tree to walk. Degrade to a simpler heuristic instead
            # of giving up entirely.
            return self._simple_verb_fallback(text_clean, original_text)

        intent_type: Optional[IntentType] = None
        verb_token = None
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in _VERB_INTENT_MAP and (token.pos_ in ("VERB", "AUX") or token.i == 0):
                intent_type = _VERB_INTENT_MAP[lemma]
                verb_token = token
                break

        if intent_type is None:
            return None

        noun_chunks = list(doc.noun_chunks)

        def chunk_text_excluding_folder_word(chunk) -> str:
            words = [t.text for t in chunk if t.text.lower() not in _FOLDER_NOUNS and t.pos_ != "DET"]
            return " ".join(words).strip()

        destination_text: Optional[str] = None
        object_text: Optional[str] = None
        mentions_folder = False

        for chunk in noun_chunks:
            head_prep = chunk.root.head
            is_dest = (
                head_prep.pos_ == "ADP" and head_prep.text.lower() in _DEST_PREPOSITIONS
            )
            cleaned = chunk_text_excluding_folder_word(chunk)
            if any(t.text.lower() in _FOLDER_NOUNS for t in chunk):
                mentions_folder = True
            if is_dest and destination_text is None and cleaned:
                destination_text = cleaned
            elif object_text is None and cleaned:
                object_text = cleaned

        parsed = ParsedIntent(intent=intent_type, text=original_text)

        if intent_type == IntentType.CREATE_FILE and mentions_folder:
            parsed.intent = IntentType.CREATE_FOLDER

        if parsed.intent in (IntentType.RENAME,):
            # "rename X to Y" — object_text is the source, destination_text is new name
            parsed.source = _clean_extracted(object_text, is_source=True)
            parsed.filename = _clean_extracted(destination_text)
        elif parsed.intent in (IntentType.MOVE, IntentType.COPY):
            parsed.source = _clean_extracted(object_text, is_source=True)
            parsed.destination = _clean_extracted(destination_text)
        elif parsed.intent in (IntentType.NAVIGATE,):
            parsed.destination = _clean_extracted(destination_text or object_text)
        elif parsed.intent in (IntentType.LIST,):
            pass
        else:
            parsed.filename = _clean_extracted(object_text)
            parsed.destination = _clean_extracted(destination_text)

        parsed.file_type = self._extract_file_type(text_clean)
        parsed.folder = self._extract_folder(text_clean, parsed.intent)
        parsed.date_expr = self._extract_date(text_clean)
        parsed.keywords = self._build_keywords(parsed)

        # Require that we actually recovered something actionable before
        # trusting this fallback parse — an intent with no filename/dest for
        # an action that needs one is more likely noise than a real command.
        needs_target = parsed.intent not in (IntentType.LIST,)
        has_target = bool(parsed.filename or parsed.source or parsed.destination)
        if needs_target and not has_target:
            return None

        parsed.confidence = 0.65
        parsed.parse_method = "spacy_fallback"
        logger.debug(
            f"spaCy fallback parse: intent={parsed.intent.name}, filename={parsed.filename}, "
            f"source={parsed.source}, dest={parsed.destination}"
        )
        return parsed

    def _simple_verb_fallback(self, text_clean: str, original_text: str) -> Optional[ParsedIntent]:
        """Last-resort heuristic used when the trained spaCy pipeline
        (en_core_web_sm) isn't installed, so no lemmatizer/dependency parse is
        available. Looks only at the first word against the same verb→intent
        map the full spaCy fallback uses, and treats the rest of the sentence
        as the filename/destination. Cruder than the dependency-parse path
        (no lemmatization, so only exact verb forms match), but still better
        than dropping straight to UNKNOWN."""
        words = text_clean.split()
        if not words:
            return None
        intent_type = _VERB_INTENT_MAP.get(words[0])
        if intent_type is None:
            return None

        remainder = " ".join(words[1:])
        stripped, dest = self._strip_trailing_location(remainder)

        parsed = ParsedIntent(intent=intent_type, text=original_text)
        if intent_type == IntentType.RENAME:
            parsed.source = stripped
            parsed.filename = dest
        elif intent_type in (IntentType.MOVE, IntentType.COPY):
            parsed.source = stripped
            parsed.destination = dest
        elif intent_type == IntentType.NAVIGATE:
            parsed.destination = dest or stripped
        elif intent_type == IntentType.LIST:
            pass
        else:
            parsed.filename = stripped
            parsed.destination = dest

        if intent_type == IntentType.CREATE_FILE and any(w in _FOLDER_NOUNS for w in words):
            parsed.intent = IntentType.CREATE_FOLDER
            if parsed.filename:
                parsed.filename = " ".join(
                    w for w in parsed.filename.split() if w not in _FOLDER_NOUNS
                )

        needs_target = parsed.intent not in (IntentType.LIST,)
        has_target = bool(parsed.filename or parsed.source or parsed.destination)
        if needs_target and not has_target:
            return None

        parsed.file_type = self._extract_file_type(text_clean)
        parsed.folder = self._extract_folder(text_clean, parsed.intent)
        parsed.date_expr = self._extract_date(text_clean)
        parsed.keywords = self._build_keywords(parsed)
        parsed.confidence = 0.5
        parsed.parse_method = "keyword_verb_fallback"
        logger.debug(
            f"Simple verb fallback parse (no spaCy model): intent={parsed.intent.name}, "
            f"filename={parsed.filename}, source={parsed.source}, dest={parsed.destination}"
        )
        return parsed

    def _keyword_search_fallback(self, text_clean: str, original_text: str) -> ParsedIntent:
        parsed = ParsedIntent(intent=IntentType.UNKNOWN, text=original_text)
        if any(
            re.search(rf"\b{w}\b", text_clean)
            for w in ["find", "search", "locate", "look"]
        ):
            parsed.intent = IntentType.SEARCH
            parsed.filename = text_clean
            parsed.file_type = self._extract_file_type(text_clean)
            parsed.date_expr = self._extract_date(text_clean)
            parsed.keywords = [w for w in text_clean.split() if len(w) > 2]
            parsed.confidence = 0.4
            parsed.parse_method = "keyword_fallback"
        return parsed

    def parse(self, text: str) -> ParsedIntent:
        if not text:
            return ParsedIntent(intent=IntentType.UNKNOWN, text=text)

        # Strip ASR sentence-ending punctuation ("...on desktop.") before any
        # pattern matching — otherwise "desktop." never matches known folder
        # names, gets treated as a literal (wrong) destination.
        text_clean = _strip_trailing_punctuation(text.strip().lower())

        parsed = self._regex_parse(text_clean, text)
        if parsed is not None:
            return parsed

        parsed = self._spacy_fallback_parse(text_clean, text)
        if parsed is not None:
            return parsed

        return self._keyword_search_fallback(text_clean, text)