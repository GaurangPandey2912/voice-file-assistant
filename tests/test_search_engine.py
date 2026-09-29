from pathlib import Path

from voice_assistant.search.engine import SearchEngine


def _entry(name, folder="Documents", abs_path=None, size=100):
    abs_path = abs_path or f"/home/user/{folder}/{name}"
    return {
        "filename": name,
        "absolute_path": abs_path,
        "extension": Path(name).suffix.lower(),
        "size": size,
        "created_date": None,
        "modified_date": None,
        "folder": folder,
    }


def _engine(index):
    # use_semantic=False keeps these tests fast and independent of any
    # downloaded embedding model — semantic scoring is covered separately.
    return SearchEngine(index, use_semantic=False)


def test_exact_match_ranks_first():
    index = [
        _entry("networking notes.txt"),
        _entry("random file.txt"),
        _entry("networking.txt"),
    ]
    engine = _engine(index)
    results = engine.search("networking notes.txt")
    assert results[0].name == "networking notes.txt"
    assert "exact" in results[0].match_type


def test_token_match_and_ranking_order():
    index = [
        _entry("quarterly invoice.pdf"),
        _entry("resume.docx"),
        _entry("invoice draft.pdf"),
    ]
    engine = _engine(index)
    results = engine.search("invoice")
    names = [r.name for r in results]
    assert "quarterly invoice.pdf" in names
    assert "invoice draft.pdf" in names
    assert "resume.docx" not in names[:2]


def test_file_type_filter_boosts_matching_extension():
    index = [
        _entry("report.pdf"),
        _entry("report.docx"),
    ]
    engine = _engine(index)
    results = engine.search("report pdf")
    assert results[0].name == "report.pdf"


def test_current_directory_context_boost():
    index = [
        _entry("notes.txt", folder="Documents", abs_path="/home/user/Documents/notes.txt"),
        _entry("notes.txt", folder="Desktop", abs_path="/home/user/Desktop/notes.txt"),
    ]
    engine = _engine(index)
    # Without context both are tied (same name/extension), so this only
    # verifies context breaks the tie in favor of the current directory.
    results = engine.search("notes", current_dir=Path("/home/user/Desktop"))
    assert results[0].folder == "Desktop"
    assert "current_dir" in results[0].match_type


def test_recent_interaction_context_boost():
    # Neither filename exactly matches the query stem, so this isolates the
    # recency signal instead of being decided by the exact-match bonus.
    index = [
        _entry("assignment draft.docx", abs_path="/home/user/Documents/assignment_draft.docx"),
        _entry("assignment final.docx", abs_path="/home/user/Documents/assignment_final.docx"),
    ]
    engine = _engine(index)
    recent = [Path("/home/user/Documents/assignment_final.docx")]
    results = engine.search("assignment", recent_paths=recent)
    assert results[0].name == "assignment final.docx"
    assert "recent" in results[0].match_type


def test_empty_index_returns_no_results():
    engine = _engine([])
    assert engine.search("anything") == []


def test_update_index_replaces_candidates():
    engine = _engine([_entry("old.txt")])
    engine.update_index([_entry("new.txt")])
    results = engine.search("new")
    assert any(r.name == "new.txt" for r in results)
    assert not any(r.name == "old.txt" for r in results)
