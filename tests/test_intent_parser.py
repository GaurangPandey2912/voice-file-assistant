from voice_assistant.intent.parser import IntentParser, IntentType


def test_create_file_regex():
    parser = IntentParser()
    result = parser.parse("create file project notes")
    assert result.intent == IntentType.CREATE_FILE
    assert result.filename == "project notes"
    assert result.parse_method == "regex"
    assert result.confidence == 1.0


def test_create_folder_regex():
    parser = IntentParser()
    result = parser.parse("make a folder called Networking")
    assert result.intent == IntentType.CREATE_FOLDER
    assert "networking" in result.filename.lower()


def test_delete_regex():
    parser = IntentParser()
    result = parser.parse("delete report.pdf")
    assert result.intent == IntentType.DELETE
    assert result.filename == "report.pdf"
    assert result.is_destructive


def test_rename_regex():
    parser = IntentParser()
    result = parser.parse("rename report to final report")
    assert result.intent == IntentType.RENAME
    assert result.source == "report"
    assert result.filename == "final report"


def test_move_regex():
    parser = IntentParser()
    result = parser.parse("move report.pdf to Documents")
    assert result.intent == IntentType.MOVE
    assert result.source == "report.pdf"
    assert result.destination == "documents"
    assert result.needs_confirmation


def test_navigate_regex():
    parser = IntentParser()
    result = parser.parse("go to Downloads")
    assert result.intent == IntentType.NAVIGATE
    assert result.destination == "downloads"


def test_search_regex():
    parser = IntentParser()
    result = parser.parse("find networking notes")
    assert result.intent == IntentType.SEARCH
    assert result.filename == "networking notes"


def test_search_with_date_extraction():
    parser = IntentParser()
    result = parser.parse("find yesterday's networking notes")
    assert result.intent == IntentType.SEARCH
    assert result.date_expr is not None


def test_search_with_file_type():
    parser = IntentParser()
    result = parser.parse("list all pdf files")
    assert result.intent == IntentType.LIST
    assert result.file_type == "pdf"


def test_spacy_fallback_handles_unusual_phrasing():
    """A phrasing the regex patterns don't cover ('erase' instead of
    'delete'/'remove'/'trash') should still be picked up — either via the full
    spaCy dependency-parse fallback (if en_core_web_sm is installed) or the
    simpler first-word verb fallback (if it isn't) — at reduced confidence."""
    parser = IntentParser()
    result = parser.parse("erase old resume")
    assert result.intent == IntentType.DELETE
    assert result.parse_method in ("regex", "spacy_fallback", "keyword_verb_fallback")
    if result.parse_method != "regex":
        assert result.is_low_confidence


def test_unknown_command():
    parser = IntentParser()
    result = parser.parse("what is the weather today")
    # Should not be confidently mapped to a destructive or file-mutating intent.
    assert result.intent in (IntentType.UNKNOWN, IntentType.SEARCH)
    if result.intent != IntentType.UNKNOWN:
        assert result.is_low_confidence


def test_simple_verb_fallback_without_spacy_model():
    """Directly exercises the no-model fallback path (as if en_core_web_sm
    weren't installed), independent of whatever spaCy setup the test machine
    actually has."""
    parser = IntentParser()
    result = parser._simple_verb_fallback("erase old resume", "erase old resume")
    assert result is not None
    assert result.intent == IntentType.DELETE
    assert result.filename == "old resume"
    assert result.parse_method == "keyword_verb_fallback"
    assert result.is_low_confidence


def test_trailing_whisper_punctuation_is_stripped():
    """Whisper punctuates transcripts like written sentences ('...on
    desktop.') — a trailing period must not stop 'desktop.' from resolving
    to the known 'desktop' folder."""
    parser = IntentParser()
    result = parser.parse("Create a folder called ABC on desktop.")
    assert result.intent == IntentType.CREATE_FOLDER
    assert result.destination == "desktop"
    assert not result.destination.endswith(".")


def test_move_command_strips_filler_words_and_from_clause():
    """'the file X from desktop' should resolve to just 'X' — the filler
    words and redundant location clause aren't part of the actual filename."""
    parser = IntentParser()
    result = parser.parse("Move the file xyz.py from desktop to the folder abc")
    assert result.intent == IntentType.MOVE
    assert result.source == "xyz.py"
    assert result.destination == "abc"


def test_file_type_extraction_ignores_generic_words():
    """'a file' / 'the files' shouldn't be mistaken for a file type/extension
    named 'a' or 'the' — only real type words (pdf, python, etc.) should."""
    parser = IntentParser()
    result = parser.parse("create a file called xyz.py on desktop")
    assert result.file_type is None

    result2 = parser.parse("list all pdf files")
    assert result2.file_type == "pdf"



def test_empty_text():
    parser = IntentParser()
    result = parser.parse("")
    assert result.intent == IntentType.UNKNOWN
