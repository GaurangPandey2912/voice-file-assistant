"""
Generates a labeled dataset of voice-command utterances for training and
evaluating an intent classifier, matching the IntentType categories used by
voice_assistant/intent/parser.py.

Approach: template-based synthetic generation. Each intent has several
sentence templates with {placeholders} (filename, folder, location, date)
filled from vocabulary lists via random sampling. This is the standard way
to bootstrap an intent-classification dataset when you don't have real
labeled user logs yet — it's a reasonable, defensible choice to describe in
a report's "Dataset" section, and can later be supplemented with real
transcribed commands from actual usage of the assistant.

An UNKNOWN / out-of-scope class is included so the classifier also learns to
reject commands that aren't file-management related, matching how the app
needs to behave in practice.

Usage:
    python generate_dataset.py [--per-class N] [--seed 42]

Outputs (in ./data/):
    dataset_full.csv          all generated (text, label) pairs
    train.csv / val.csv / test.csv   stratified 70/15/15 split
    label_list.json           ordered list of label names (id -> label)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent / "data"

LABELS = [
    "CREATE_FILE",
    "CREATE_FOLDER",
    "DELETE",
    "DUPLICATE",
    "RENAME",
    "MOVE",
    "COPY",
    "OPEN",
    "NAVIGATE",
    "LIST",
    "SEARCH",
    "UNKNOWN",
]

# -- vocabulary ---------------------------------------------------------

FILES = [
    "report.pdf", "resume.docx", "notes.txt", "budget.xlsx", "photo.jpg",
    "invoice.pdf", "presentation.pptx", "video.mp4", "data.csv",
    "config.json", "thesis draft", "assignment.docx", "lecture notes",
    "project proposal", "meeting minutes", "vacation photos",
    "tax documents", "song.mp3", "screenshot.png", "backup.zip",
    "essay.docx", "timetable.xlsx", "contract.pdf", "syllabus.pdf",
    "networking notes", "research paper", "internship report",
    "code review notes", "final draft", "grocery list", "travel itinerary",
]

FILES_ALT = [
    "final report.pdf", "updated resume.docx", "class notes.txt",
    "quarterly budget.xlsx", "holiday photo.jpg", "march invoice.pdf",
    "sales presentation.pptx", "demo video.mp4", "sensor data.csv",
    "app config.json", "final thesis.docx", "homework.docx",
    "revised notes", "new proposal", "team minutes", "beach photos",
    "2025 tax forms", "favorite song.mp3", "bug screenshot.png",
    "old backup.zip", "final essay.docx", "exam timetable.xlsx",
]

FOLDERS = [
    "Networking", "Semester 6", "College Work", "Photos", "Personal",
    "Work", "Backup", "Old Files", "Assignments", "Research", "Projects",
    "Invoices", "Media", "Archive", "Shared", "Drafts",
]

KNOWN_LOCATIONS = ["desktop", "documents", "downloads", "home"]

DATE_PHRASES = [
    "yesterday", "today", "last week", "last month", "two days ago",
    "last Monday", "this morning", "last year", "a few days ago",
]

FILE_TYPES = ["pdf", "python", "text", "image", "document", "markdown"]


def _rand(seq):
    return random.choice(seq)


# -- templates ------------------------------------------------------------
# Each template is a callable producing one filled sentence.

def _create_file_templates():
    f = _rand(FILES)
    loc = _rand(KNOWN_LOCATIONS)
    folder = _rand(FOLDERS)
    return _rand([
        f"create a file called {f}",
        f"create file {f}",
        f"make a new file named {f}",
        f"add a file {f}",
        f"please create {f}",
        f"new file {f} on {loc}",
        f"create {f} in {folder}",
        f"make a file {f} on {loc}",
        f"can you create a file called {f}",
    ])


def _create_folder_templates():
    folder = _rand(FOLDERS)
    loc = _rand(KNOWN_LOCATIONS)
    return _rand([
        f"create a folder called {folder}",
        f"make a new folder named {folder}",
        f"create folder {folder} on {loc}",
        f"add a directory called {folder}",
        f"please make a folder {folder}",
        f"can you create a folder named {folder}",
        f"new folder {folder}",
    ])


def _delete_templates():
    f = _rand(FILES)
    folder = _rand(FOLDERS)
    return _rand([
        f"delete {f}",
        f"remove {f}",
        f"trash {f}",
        f"erase {f}",
        f"please delete the file {f}",
        f"get rid of {f}",
        f"delete the folder {folder}",
        f"can you delete {f}",
        f"remove the file {f} for me",
    ])


def _duplicate_templates():
    f = _rand(FILES)
    return _rand([
        f"duplicate {f}",
        f"make a copy of {f}",
        f"create a copy of {f}",
        f"clone {f}",
        f"duplicate the file {f}",
        f"can you make a copy of {f}",
    ])


def _rename_templates():
    f = _rand(FILES)
    f2 = _rand(FILES_ALT)
    return _rand([
        f"rename {f} to {f2}",
        f"rename the file {f} to {f2}",
        f"change the name of {f} to {f2}",
        f"please rename {f} to {f2}",
    ])


def _move_templates():
    f = _rand(FILES)
    folder = _rand(FOLDERS)
    loc = _rand(KNOWN_LOCATIONS)
    return _rand([
        f"move {f} to {folder}",
        f"move {f} to {loc}",
        f"shift {f} into {folder}",
        f"relocate {f} to {folder}",
        f"move the file {f} to {folder}",
        f"can you move {f} to {loc}",
    ])


def _copy_templates():
    f = _rand(FILES)
    folder = _rand(FOLDERS)
    loc = _rand(KNOWN_LOCATIONS)
    return _rand([
        f"copy {f} to {folder}",
        f"copy {f} to {loc}",
        f"copy the file {f} to {folder}",
        f"please copy {f} to {loc}",
    ])


def _open_templates():
    f = _rand(FILES)
    return _rand([
        f"open {f}",
        f"launch {f}",
        f"open the file {f}",
        f"please open {f}",
        f"start {f}",
        f"can you open {f}",
    ])


def _navigate_templates():
    loc = _rand(KNOWN_LOCATIONS)
    folder = _rand(FOLDERS)
    return _rand([
        f"go to {loc}",
        f"navigate to {loc}",
        f"go to {folder}",
        f"change directory to {loc}",
        f"take me to {loc}",
        "go home",
        "go back",
        "go up",
        f"navigate to the {folder} folder",
        f"please go to {loc}",
        f"open the {folder} folder",
        f"switch to {loc}",
        f"go into {folder}",
        f"head to {loc}",
        f"show me {loc}",
        f"can you go to {folder}",
        f"navigate into the {folder} directory",
        f"take me back to {loc}",
        f"go to the {folder} folder please",
    ])


def _list_templates():
    ftype = _rand(FILE_TYPES)
    folder = _rand(FOLDERS)
    return _rand([
        "list all files",
        "show all folders",
        f"list {ftype} files",
        "display all documents",
        "show me all images",
        "list files",
        "what files are here",
        f"show {ftype} files",
        "list everything in this folder",
        f"list all files in {folder}",
        f"show me the {ftype} files",
        "what's in this folder",
        "show contents of this folder",
        "display everything here",
        "list items",
        "what folders are here",
        f"show all {ftype} documents",
        "list all the files in this directory",
        f"show me what's in {folder}",
        "what's inside this folder",
        f"list all {ftype} files in {folder}",
        f"show {ftype} files in {folder}",
        f"list every {ftype} file here",
        f"can you list the {ftype} files",
    ])


def _search_templates():
    f = _rand(FILES)
    date = _rand(DATE_PHRASES)
    return _rand([
        f"find {f}",
        f"search for {f}",
        f"locate {f}",
        f"look for {f}",
        f"find {f} from {date}",
        f"search for the {f} I made {date}",
        "find all invoices",
        f"look for my {f}",
        f"can you find {f}",
    ])


_UNKNOWN_UTTERANCES = [
    "what is the weather today",
    "tell me a joke",
    "how are you",
    "what time is it",
    "play some music",
    "what's the capital of France",
    "who is the president",
    "how do I cook pasta",
    "remind me to call mom",
    "set an alarm for seven a m",
    "what's on my calendar",
    "turn off the lights",
    "good morning",
    "thank you",
    "stop listening",
    "how far is the moon",
    "sing me a song",
    "what's two plus two",
    "translate hello to french",
    "tell me the news",
    "how do I fix my wifi",
    "who won the match yesterday",
    "what's a good movie to watch",
    "can you help me relax",
    "increase the volume",
]


_TOPICS = [
    "cricket", "politics", "AI", "space", "history", "music", "cooking",
    "travel", "health", "technology", "art", "movies", "sports", "science",
    "the stock market", "philosophy", "gardening", "fashion",
    "photography", "climate change", "football", "astronomy", "economics",
]

_ACTIVITIES = [
    "lose weight", "learn guitar", "make coffee", "fix a flat tire",
    "meditate", "improve my memory", "learn to swim", "write a resume",
    "start a business", "get better sleep", "learn python",
    "plan a trip", "bake a cake", "study for exams", "manage stress",
]


def _unknown_templates():
    choice = random.random()
    if choice < 0.4:
        return _rand(_UNKNOWN_UTTERANCES)
    elif choice < 0.65:
        return _rand([
            f"what is {_rand(_TOPICS)}",
            f"tell me about {_rand(_TOPICS)}",
            f"what do you think about {_rand(_TOPICS)}",
        ])
    else:
        return _rand([
            f"how do i {_rand(_ACTIVITIES)}",
            f"can you help me {_rand(_ACTIVITIES)}",
            f"any tips to {_rand(_ACTIVITIES)}",
        ])


_GENERATORS = {
    "CREATE_FILE": _create_file_templates,
    "CREATE_FOLDER": _create_folder_templates,
    "DELETE": _delete_templates,
    "DUPLICATE": _duplicate_templates,
    "RENAME": _rename_templates,
    "MOVE": _move_templates,
    "COPY": _copy_templates,
    "OPEN": _open_templates,
    "NAVIGATE": _navigate_templates,
    "LIST": _list_templates,
    "SEARCH": _search_templates,
    "UNKNOWN": _unknown_templates,
}


def generate(per_class: int, seed: int) -> list[tuple[str, str]]:
    random.seed(seed)
    rows: list[tuple[str, str]] = []
    for label in LABELS:
        gen_fn = _GENERATORS[label]
        seen: set[str] = set()
        attempts = 0
        # Oversample and dedupe so we still hit per_class even if the
        # template space produces some repeats via random sampling.
        while len(seen) < per_class and attempts < per_class * 20:
            text = gen_fn()
            attempts += 1
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, label))
    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=120,
                         help="Target number of examples per intent class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = generate(args.per_class, args.seed)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    full_path = OUTPUT_DIR / "dataset_full.csv"
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    # Stratified 70/15/15 split so every class is represented proportionally
    # in train/val/test — important with 12 classes and a modest dataset size.
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=args.seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, random_state=args.seed,
        stratify=temp_labels,
    )

    for name, t_texts, t_labels in [
        ("train", train_texts, train_labels),
        ("val", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ]:
        path = OUTPUT_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(zip(t_texts, t_labels))
        print(f"Wrote {len(t_texts):4d} examples -> {path}")

    with open(OUTPUT_DIR / "label_list.json", "w", encoding="utf-8") as f:
        json.dump(LABELS, f, indent=2)

    print(f"\nTotal examples: {len(rows)}")
    print(f"Classes: {len(LABELS)}")
    print(f"Full dataset -> {full_path}")
    print(f"Label list -> {OUTPUT_DIR / 'label_list.json'}")


if __name__ == "__main__":
    main()