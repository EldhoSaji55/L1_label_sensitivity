from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# PROJECT PATHS AND EXPERIMENT SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE = PROJECT_ROOT / "data" / "reviewed_data" / "chosen_data.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "reviewed_data"

DEFAULT_LABELS = ("German", "Malayalam", "Arabic", "Unknown")
SEED = 42
OUTPUT_FIELDS = ["S_No", "sentence_id", "language", "Sentence"]


# COMMAND-LINE OPTIONS

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LABELS),
        metavar="LANGUAGE",
        help=(
            "L1 conditions to create. Keep Unknown as the no-L1 baseline. "
            "Default: German Malayalam Arabic Unknown"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


#   Clean labels, reject duplicates, and preserve the supplied order.
   
def prepare_labels(raw_labels: list[str]) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()

    for raw_label in raw_labels:
        label = raw_label.strip()
        if not label:
            raise ValueError("Language labels cannot be empty.")

        comparison_key = label.casefold()
        if comparison_key == "unknown":
            label = "Unknown"
        if comparison_key in seen:
            raise ValueError(f"Duplicate language label: {label}")

        seen.add(comparison_key)
        labels.append(label)

    if len(labels) < 2:
        raise ValueError("Provide at least two language conditions.")
    if "unknown" not in seen:
        raise ValueError("Include 'Unknown' as the no-L1 baseline condition.")

    return tuple(labels)


# VERIFIED-REFERENCE READING AND VALIDATION

def read_reference(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Verified reference file not found. Place it at "
            f"{path}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("The reference file contains no rows.")

    columns = set(reader.fieldnames or [])
    if "sentence_id" not in columns:
        raise ValueError("The reference file is missing the 'sentence_id' column.")

    if "original" in columns:
        sentence_column = "original"
    elif "Sentence" in columns:
        sentence_column = "Sentence"
    else:
        raise ValueError("The reference needs an 'original' or 'Sentence' column.")

    seen: set[str] = set()
    clean: list[dict[str, str]] = []

    for row in rows:
        sentence_id = row["sentence_id"].strip()
        sentence = row[sentence_column].strip()

        if not sentence_id or not sentence:
            raise ValueError("Reference rows cannot have empty IDs or sentences.")
        if sentence_id in seen:
            raise ValueError(f"Duplicate sentence_id: {sentence_id}")

        seen.add(sentence_id)
        clean.append({"sentence_id": sentence_id, "Sentence": sentence})

    return clean


# FOUR-CONDITION JOB GENERATION

def build_jobs(
    reference_rows: list[dict[str, str]],
    labels: tuple[str, ...] = DEFAULT_LABELS,
    seed: int = SEED,
) -> list[dict[str, object]]:
    jobs = [
        {
            "sentence_id": row["sentence_id"],
            "language": label,
            "Sentence": row["Sentence"],
        }
        for row in reference_rows
        for label in labels
    ]
    random.Random(seed).shuffle(jobs)
    return [{"S_No": index, **job} for index, job in enumerate(jobs, 1)]


# GENERATED-JOB VALIDATION

def validate_jobs(
    jobs: list[dict[str, object]],
    sentence_count: int,
    labels: tuple[str, ...],
) -> None:
    expected_jobs = sentence_count * len(labels)
    if len(jobs) != expected_jobs:
        raise ValueError(f"Expected {expected_jobs} jobs, but created {len(jobs)}.")

    id_counts = Counter(str(job["sentence_id"]) for job in jobs)
    if len(id_counts) != sentence_count:
        raise ValueError("The number of unique sentence IDs changed during expansion.")
    if any(count != len(labels) for count in id_counts.values()):
        raise ValueError(
            f"Every sentence_id must occur exactly {len(labels)} times."
        )

    grouped_languages: dict[str, set[str]] = defaultdict(set)
    grouped_sentences: dict[str, set[str]] = defaultdict(set)
    for job in jobs:
        sentence_id = str(job["sentence_id"])
        grouped_languages[sentence_id].add(str(job["language"]))
        grouped_sentences[sentence_id].add(str(job["Sentence"]))

    for sentence_id in id_counts:
        if grouped_languages[sentence_id] != set(labels):
            raise ValueError(f"Incomplete language conditions for {sentence_id}.")
        if len(grouped_sentences[sentence_id]) != 1:
            raise ValueError(f"Sentence text changed across labels for {sentence_id}.")


# CSV OUTPUT

def write_jobs(path: Path, jobs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(jobs)


# MAIN 

def main() -> int:
    args = parse_arguments()

    try:
        labels = prepare_labels(args.languages)
        rows = read_reference(REFERENCE_FILE)
        jobs = build_jobs(rows, labels=labels, seed=args.seed)
        validate_jobs(jobs, sentence_count=len(rows), labels=labels)
        output_file = OUTPUT_DIRECTORY / f"chosen_data_{len(jobs)}.csv"
        write_jobs(output_file, jobs)
    except (FileNotFoundError, ValueError, UnicodeError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Verified sentences: {len(rows)}")
    print(f"Language conditions ({len(labels)}): {', '.join(labels)}")
    print(f"Calculation: {len(rows)} x {len(labels)} = {len(jobs)}")
    print(f"Created jobs: {len(jobs)}")
    print(f"Saved: {output_file}")
    return 0


# SCRIPT ENTRY POINT

if __name__ == "__main__":
    raise SystemExit(main())
