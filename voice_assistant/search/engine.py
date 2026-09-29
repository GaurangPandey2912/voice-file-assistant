from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from voice_assistant.search.semantic import SemanticEmbedder

logger = logging.getLogger("voice_assistant.search")

# Weights for combining retrieval signals into one relevance score. Kept as
# module-level constants so the ranking behaviour is easy to tune and to cite
# directly in a report (e.g. "semantic similarity contributes up to 35 points").
_WEIGHT_EXACT = 100.0
_WEIGHT_TOKEN = 15.0
_WEIGHT_FUZZY = 25.0
_WEIGHT_FOLDER_TOKEN = 10.0
_WEIGHT_FILETYPE = 20.0
_WEIGHT_EXTENSION = 30.0
_WEIGHT_SEMANTIC = 35.0
_SEMANTIC_MIN_SIMILARITY = 0.30  # below this, treat the semantic signal as noise
_WEIGHT_CURRENT_DIR = 15.0
_WEIGHT_RECENT_BASE = 12.0

# A bare filename/folder name with fewer than two real words carries too
# little text for the embedding model to compare meaningfully — short
# strings ("abz", "Video") can land at deceptively high cosine similarity
# to an unrelated query purely due to embedding-space geometry, not genuine
# relatedness. Semantic scoring is only trusted when there's a real content
# preview, or the name itself has at least two alphabetic words.
_MIN_SEMANTIC_WORDS = 2
_WORD_PATTERN = re.compile(r"[a-zA-Z]{3,}")


@dataclass
class SearchResult:
    path: Path
    name: str
    extension: str
    size: int
    created: Optional[str] = None
    modified: Optional[str] = None
    folder: str = ""
    score: float = 0.0
    match_type: str = ""


class SearchEngine:
    def __init__(self, index_data: list[dict], use_semantic: bool = True) -> None:
        self._index = index_data
        self._file_types = {
            "pdf": ".pdf",
            "python": ".py",
            "text": ".txt",
            "markdown": ".md",
            "document": ".docx",
            "image": ".jpg",
            "folder": "",
        }
        self._embedder = SemanticEmbedder() if use_semantic else None
        # absolute_path -> embedding vector, so re-searching doesn't re-encode
        # every file every time; only new/changed files pay the encode cost.
        self._embedding_cache: dict[str, "object"] = {}

    def update_index(self, index_data: list[dict]) -> None:
        self._index = index_data
        current_paths = {entry["absolute_path"] for entry in index_data}
        for stale_path in list(self._embedding_cache.keys()):
            if stale_path not in current_paths:
                del self._embedding_cache[stale_path]

    def search(
        self,
        query: str,
        max_results: int = 10,
        current_dir: Optional[Path] = None,
        recent_paths: Optional[list[Path]] = None,
    ) -> list[SearchResult]:
        """Hybrid, context-aware search.

        Combines four signals:
          1. Filename relevance — exact / token / fuzzy string matching.
          2. Semantic similarity — cosine similarity between the query and each
             file's filename + content-preview embedding (falls back gracefully
             if sentence-transformers isn't installed, and is skipped entirely
             for names too short/bare to embed meaningfully).
          3. Current-directory context — a small boost for files already in
             the folder the user is standing in.
          4. Recent-interaction context — a decaying boost for files the user
             has recently created, opened, or otherwise touched this session.
        """
        if not self._index:
            return []

        entries_by_path: dict[str, dict] = {
            entry["absolute_path"]: entry for entry in self._index
        }
        candidates: list[SearchResult] = []
        seen_paths: set[str] = set()

        for entry in self._index:
            abs_path = entry["absolute_path"]
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            candidates.append(
                SearchResult(
                    path=Path(abs_path),
                    name=entry["filename"],
                    extension=entry["extension"],
                    size=entry.get("size", 0),
                    created=entry.get("created_date"),
                    modified=entry.get("modified_date"),
                    folder=entry.get("folder", ""),
                )
            )

        recent_rank: dict[str, int] = {}
        if recent_paths:
            for i, p in enumerate(recent_paths):
                recent_rank.setdefault(str(Path(p).resolve()), i)

        scored = self._score_candidates(
            candidates, query, entries_by_path, current_dir, recent_rank
        )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:max_results]

    # -- keyword / fuzzy / metadata scoring ---------------------------------

    def _score_candidates(
        self,
        candidates: list[SearchResult],
        query: str,
        entries_by_path: dict[str, dict],
        current_dir: Optional[Path],
        recent_rank: dict[str, int],
    ) -> list[SearchResult]:
        query_lower = query.lower().strip()
        query_terms = self._query_terms(query_lower)

        semantic_scores = self._semantic_scores(candidates, query, entries_by_path)
        current_dir_resolved = str(current_dir.resolve()) if current_dir else None

        for result in candidates:
            score = 0.0
            match_types: list[str] = []

            # Resolve once per candidate and reuse for both context checks below.
            # Comparing a resolved current_dir against an *unresolved* result path
            # (or vice versa) silently never matches on Windows, where resolve()
            # normalizes drive letters/separators — always compare like-for-like.
            resolved_path = result.path.resolve()
            resolved_parent_str = str(resolved_path.parent)
            resolved_path_str = str(resolved_path)

            name_lower = result.name.lower()
            stem_lower = Path(result.name).stem.lower()

            if query_lower == name_lower or query_lower == stem_lower:
                score += _WEIGHT_EXACT
                match_types.append("exact")

            if result.extension:
                ext_lower = result.extension.lower().lstrip(".")
                for ftype_key, ftype_val in self._file_types.items():
                    if ftype_val.lstrip(".") == ext_lower or ftype_key == ext_lower:
                        if ftype_key in query_lower or ext_lower in query_lower:
                            score += _WEIGHT_FILETYPE
                            if "filetype" not in match_types:
                                match_types.append("filetype")

            for token in query_terms:
                if token in name_lower:
                    score += _WEIGHT_TOKEN
                    if "token" not in match_types:
                        match_types.append("token")
                sim = SequenceMatcher(None, token, stem_lower).ratio()
                if sim > 0.6:
                    score += sim * _WEIGHT_FUZZY
                    if "fuzzy" not in match_types:
                        match_types.append("fuzzy")

            if result.folder:
                folder_lower = result.folder.lower()
                for token in query_terms:
                    if token in folder_lower or folder_lower in token:
                        score += _WEIGHT_FOLDER_TOKEN
                        if "folder" not in match_types:
                            match_types.append("folder")

            if result.extension:
                ext_query = re.search(r"\.(\w+)", query_lower)
                if ext_query and ext_query.group(1) == result.extension.lstrip("."):
                    score += _WEIGHT_EXTENSION
                    if "extension" not in match_types:
                        match_types.append("extension")

            sem_score = semantic_scores.get(str(result.path))
            if sem_score is not None and sem_score >= _SEMANTIC_MIN_SIMILARITY:
                score += sem_score * _WEIGHT_SEMANTIC
                match_types.append("semantic")

            if current_dir_resolved and resolved_parent_str == current_dir_resolved:
                score += _WEIGHT_CURRENT_DIR
                match_types.append("current_dir")

            rank = recent_rank.get(resolved_path_str)
            if rank is not None:
                score += _WEIGHT_RECENT_BASE / (rank + 1)
                match_types.append("recent")

            result.score = round(score, 2)
            result.match_type = "+".join(match_types) if match_types else "partial"

        return candidates

    def _query_terms(self, query_lower: str) -> list[str]:
        query_tokens = [t for t in re.split(r"[\s,]+", query_lower) if len(t) > 1]
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "find", "search",
            "locate", "look", "for", "from", "to", "in", "of", "all", "my",
            "about", "with", "that", "this", "and", "or", "but",
        }
        return [t for t in query_tokens if t not in stop_words]

    # -- semantic similarity --------------------------------------------------

    def _entry_text(self, entry: dict) -> str:
        parts = [entry.get("filename", "")]
        preview = entry.get("content_preview")
        if preview:
            parts.append(preview[:300])
        return " ".join(parts).strip()

    def _has_enough_text_for_semantics(self, entry: dict) -> bool:
        """Guards against spurious high-similarity matches on bare short
        names (a folder called 'abz', a file called 'Video') by requiring
        either real content, or at least two real words in the filename."""
        if entry.get("content_preview"):
            return True
        word_count = len(_WORD_PATTERN.findall(entry.get("filename", "")))
        return word_count >= _MIN_SEMANTIC_WORDS

    def _semantic_scores(
        self,
        candidates: list[SearchResult],
        query: str,
        entries_by_path: dict[str, dict],
    ) -> dict[str, float]:
        """Returns {absolute_path: cosine_similarity} for every eligible
        candidate, computed lazily and cached per-file. Returns an empty dict
        (i.e. the semantic signal contributes nothing) if embeddings aren't
        available."""
        if self._embedder is None or not self._embedder.is_available:
            return {}

        query_embedding = self._embedder.encode_one(query)
        if query_embedding is None:
            return {}

        eligible_paths: set[str] = set()
        to_encode: list[str] = []
        to_encode_paths: list[str] = []
        for result in candidates:
            abs_path = str(result.path)
            entry = entries_by_path.get(abs_path)
            if not entry or not self._has_enough_text_for_semantics(entry):
                continue
            eligible_paths.add(abs_path)
            if abs_path in self._embedding_cache:
                continue
            text = self._entry_text(entry)
            if not text:
                continue
            to_encode.append(text)
            to_encode_paths.append(abs_path)

        if to_encode:
            embeddings = self._embedder.encode(to_encode)
            if embeddings is not None:
                for path, vec in zip(to_encode_paths, embeddings):
                    self._embedding_cache[path] = vec

        scores: dict[str, float] = {}
        for result in candidates:
            abs_path = str(result.path)
            if abs_path not in eligible_paths:
                continue
            vec = self._embedding_cache.get(abs_path)
            if vec is None:
                continue
            scores[abs_path] = self._embedder.cosine_similarity(query_embedding, vec)
        return scores