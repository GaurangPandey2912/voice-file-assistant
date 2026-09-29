from pathlib import Path

import pytest

from voice_assistant.filesystem import operations as ops


def test_create_and_rename_file(tmp_path):
    success, msg = ops.create_file("draft.txt", tmp_path)
    assert success
    assert (tmp_path / "draft.txt").exists()

    success, msg = ops.rename(tmp_path / "draft.txt", "final.txt")
    assert success
    assert (tmp_path / "final.txt").exists()
    assert not (tmp_path / "draft.txt").exists()


def test_create_file_rejects_duplicate(tmp_path):
    ops.create_file("a.txt", tmp_path)
    success, msg = ops.create_file("a.txt", tmp_path)
    assert not success
    assert "already exists" in msg.lower()


def test_create_folder(tmp_path):
    success, msg = ops.create_folder("Networking", tmp_path)
    assert success
    assert (tmp_path / "Networking").is_dir()


def test_move_and_copy(tmp_path):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    f = src_dir / "file.txt"
    f.write_text("hello")

    success, msg = ops.copy_op(f, dst_dir)
    assert success
    assert (dst_dir / "file.txt").exists()
    assert f.exists()  # original still present after copy

    success, msg = ops.move(f, dst_dir / "moved_here")
    assert success
    assert not f.exists()
    assert (dst_dir / "moved_here" / "file.txt").exists()


def test_duplicate_file_naming(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    success, msg = ops.duplicate_file("notes.txt", tmp_path)
    assert success
    assert (tmp_path / "notes copy.txt").exists()

    success, msg = ops.duplicate_file("notes.txt", tmp_path)
    assert success
    assert (tmp_path / "notes copy2.txt").exists()


def test_delete_missing_file(tmp_path):
    success, msg = ops.delete(tmp_path / "ghost.txt")
    assert not success
    assert "not found" in msg.lower()


def test_resolve_path_finds_case_insensitive(tmp_path):
    (tmp_path / "Resume.pdf").touch()
    found = ops.resolve_path("resume.pdf", tmp_path)
    assert found is not None
    assert found.name == "Resume.pdf"


def test_resolve_ambiguous_multiple_matches(tmp_path):
    (tmp_path / "invoice_jan.pdf").touch()
    (tmp_path / "invoice_feb.pdf").touch()
    matches = ops.resolve_ambiguous("invoice", tmp_path)
    assert len(matches) == 2
