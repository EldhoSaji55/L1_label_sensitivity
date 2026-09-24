from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


# PROJECT PATHS

# INPUT - OUTPUT PATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw_data" / "B.dev.gold.bea19.m2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed_data"


# REGULAR-EXPRESSION PATTERNS

URL_RE = re.compile(
    r"(?i)(?:\bhttps?://\S+|\bwww\.\S+|\b\S+\.(?:com|org|net|edu|gov|io|de)\b(?:/\S*)?)"
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{7,}\d)(?!\w)")
ADDRESS_RE = re.compile(
    r"(?i)\b\d{1,5}\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\s+"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|weg|strasse|straße)\b"
)
IDENTIFIER_RE = re.compile(
    r"(?i)\b(?:student|passport|identity|customer|account|id)\s*"
    r"(?:no\.?|number|#)?\s*[:=-]?\s*[A-Z0-9-]{5,}\b"
)
MARKUP_RE = re.compile(r"<[^>]{1,200}>|&(?:nbsp|amp|lt|gt|quot);", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"(?i)(?:<unk>|\[unk\]|\{[^{}]{1,50}\}|\[REDACTED\])")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
REPEATED_SYMBOL_RE = re.compile(r"([^\w\s])\1{4,}")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
CONTEXT_HINT_RE = re.compile(
    r"(?i)\b(?:this|that|these|those|it|they|he|she|former|latter|above|below|"
    r"previous|following|aforementioned)\b"
)
POSSIBLE_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


# DATA STRUCTURES

@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    error_type: str
    correction: str
    annotator_id: str


@dataclass
class Record:
    sentence_id: str
    block_number: int
    source_file: str
    original_tokenized: str
    original: str
    gold_tokenized: str
    gold_correction: str
    token_count: int
    status: str
    edit_count: int
    error_types: str
    possible_context_dependency: int
    possible_name_or_entity: int
    automatic_filter_reasons: str = ""


# COMMAND-LINE SETTINGS AND VALIDATION

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select reproducible candidate sentences from a BEA-style M2 file."
    )
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument(
        "--correct-share",
        type=float,
        default=0.20,
        help=(
            "Desired fraction of correct control sentences in the candidate sample "
            "(default: 0.20, giving about 30 correct and 120 incorrect out of 150)."
        ),
    )
    parser.add_argument(
        "--annotator",
        default="0",
        help="M2 annotator ID whose edits should be used (default: 0).",
    )
    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size must be at least 1.")
    if args.min_tokens < 1:
        parser.error("--min-tokens must be at least 1.")
    if args.max_tokens < args.min_tokens:
        parser.error("--max-tokens must be greater than or equal to --min-tokens.")
    if not 0.0 <= args.correct_share <= 1.0:
        parser.error("--correct-share must be between 0 and 1.")

    return args


   
# M2 BLOCK AND ANNOTATION PARSING  : Yield non-empty M2 blocks as (one-based block number, lines).

def iter_m2_blocks(path: Path) -> Iterable[tuple[int, list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    for block_number, raw_block in enumerate(re.split(r"\n\s*\n", text.strip()), 1):
        lines = [line.rstrip("\r") for line in raw_block.splitlines() if line.strip()]
        if lines:
            yield block_number, lines


def parse_edit(line: str) -> Edit:
    if not line.startswith("A "):
        raise ValueError(f"Not an M2 edit line: {line}")

    fields = line[2:].split("|||")
    if len(fields) < 6:
        raise ValueError(f"Malformed M2 edit line (expected at least 6 fields): {line}")

    span = fields[0].split()
    if len(span) != 2:
        raise ValueError(f"Malformed token span in M2 edit: {line}")

    return Edit(
        start=int(span[0]),
        end=int(span[1]),
        error_type=fields[1].strip(),
        correction=fields[2].strip(),
        annotator_id=fields[-1].strip(),
    )


def is_noop(edit: Edit) -> bool:
    return edit.error_type.lower() == "noop" or (edit.start == -1 and edit.end == -1)


# GOLD EDIT VALIDATION AND APPLICATION : Reject invalid or overlapping edits before applying them.

def validate_edits(edits: Sequence[Edit], token_count: int) -> None:
    ordered = sorted(edits, key=lambda edit: (edit.start, edit.end))
    previous_end = -1

    for edit in ordered:
        if edit.start < 0 or edit.end < 0:
            raise ValueError(f"Negative non-noop edit span: {edit}")
        if edit.start > edit.end:
            raise ValueError(f"Edit start exceeds edit end: {edit}")
        if edit.end > token_count:
            raise ValueError(f"Edit span exceeds sentence length {token_count}: {edit}")
        if edit.start < previous_end:
            raise ValueError(f"Overlapping edits are not safely applicable: {edit}")
        previous_end = max(previous_end, edit.end)


"""Apply M2 edits from right to left so token offsets remain valid."""

def apply_edits(original_tokens: Sequence[str], edits: Sequence[Edit]) -> list[str]:
    validate_edits(edits, len(original_tokens))
    corrected = list(original_tokens)

    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        if edit.correction in {"", "-NONE-"}:
            replacement: list[str] = []
        else:
            replacement = edit.correction.split()
        corrected[edit.start : edit.end] = replacement

    return corrected


# TOKEN DETOKENIZATION : Produce a readable version while retaining a tokenized version separately.

def simple_detokenize(tokens: Sequence[str]) -> str:
   
    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([’][A-Za-z]+)\b", r"\1", text)
    text = re.sub(r"\s+([\"”’])(?=\s|$|[,.!?])", r"\1", text)
    text = re.sub(r"([\"“‘])\s+", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# AUTOMATIC SENTENCE FILTERING : Return conservative, machine-detectable exclusion reasons.

def automatic_filter_reasons(text: str, tokens: Sequence[str], minimum: int, maximum: int) -> list[str]:
    
    reasons: list[str] = []
    token_count = len(tokens)

    if token_count < minimum:
        reasons.append("too_short")
    if token_count > maximum:
        reasons.append("too_long")
    if URL_RE.search(text):
        reasons.append("url")
    if EMAIL_RE.search(text):
        reasons.append("possible_pii_email")
    if PHONE_RE.search(text):
        reasons.append("possible_pii_phone")
    if ADDRESS_RE.search(text):
        reasons.append("possible_pii_address")
    if IDENTIFIER_RE.search(text):
        reasons.append("possible_pii_identifier")
    if MARKUP_RE.search(text):
        reasons.append("markup")
    if PLACEHOLDER_RE.search(text):
        reasons.append("placeholder_or_redaction")
    if CONTROL_RE.search(text):
        reasons.append("control_or_replacement_character")
    if REPEATED_SYMBOL_RE.search(text):
        reasons.append("repeated_symbols")

    alpha_words = ALPHA_WORD_RE.findall(text)
    if len(alpha_words) < 3:
        reasons.append("too_few_alphabetic_words")

    non_space_count = sum(not character.isspace() for character in text)
    alpha_count = sum(character.isalpha() for character in text)
    if non_space_count and alpha_count / non_space_count < 0.45:
        reasons.append("low_alphabetic_content")

    return sorted(set(reasons))


# SENTENCE RECORD CONSTRUCTION

def parse_records(path: Path, annotator_id: str, minimum: int, maximum: int) -> list[Record]:
    records: list[Record] = []

    for block_number, lines in iter_m2_blocks(path):
        sentence_lines = [line for line in lines if line.startswith("S ")]
        if len(sentence_lines) != 1:
            records.append(
                Record(
                    sentence_id=f"{path.stem}:{block_number:06d}",
                    block_number=block_number,
                    source_file=path.name,
                    original_tokenized="",
                    original="",
                    gold_tokenized="",
                    gold_correction="",
                    token_count=0,
                    status="excluded",
                    edit_count=0,
                    error_types="",
                    possible_context_dependency=0,
                    possible_name_or_entity=0,
                    automatic_filter_reasons="missing_or_multiple_S_lines",
                )
            )
            continue

        original_tokenized = sentence_lines[0][2:].strip()
        original_tokens = original_tokenized.split()
        original = simple_detokenize(original_tokens)

        try:
            all_edits = [parse_edit(line) for line in lines if line.startswith("A ")]
        except (ValueError, IndexError) as error:
            records.append(
                Record(
                    sentence_id=f"{path.stem}:{block_number:06d}",
                    block_number=block_number,
                    source_file=path.name,
                    original_tokenized=original_tokenized,
                    original=original,
                    gold_tokenized="",
                    gold_correction="",
                    token_count=len(original_tokens),
                    status="excluded",
                    edit_count=0,
                    error_types="",
                    possible_context_dependency=int(bool(CONTEXT_HINT_RE.search(original))),
                    possible_name_or_entity=int(bool(POSSIBLE_NAME_RE.search(original))),
                    automatic_filter_reasons=f"malformed_annotation:{type(error).__name__}",
                )
            )
            continue

        annotators_present = {edit.annotator_id for edit in all_edits}
        if all_edits and annotator_id not in annotators_present:
            records.append(
                Record(
                    sentence_id=f"{path.stem}:{block_number:06d}",
                    block_number=block_number,
                    source_file=path.name,
                    original_tokenized=original_tokenized,
                    original=original,
                    gold_tokenized="",
                    gold_correction="",
                    token_count=len(original_tokens),
                    status="excluded",
                    edit_count=0,
                    error_types="",
                    possible_context_dependency=int(bool(CONTEXT_HINT_RE.search(original))),
                    possible_name_or_entity=int(bool(POSSIBLE_NAME_RE.search(original))),
                    automatic_filter_reasons="selected_annotator_missing",
                )
            )
            continue

        selected_edits = [edit for edit in all_edits if edit.annotator_id == annotator_id]
        real_edits = [edit for edit in selected_edits if not is_noop(edit)]

        reasons = automatic_filter_reasons(original, original_tokens, minimum, maximum)
        try:
            corrected_tokens = apply_edits(original_tokens, real_edits)
        except ValueError:
            corrected_tokens = []
            reasons.append("invalid_or_overlapping_gold_edits")

        status = "incorrect" if real_edits else "correct"
        error_types = ";".join(sorted({edit.error_type for edit in real_edits}))

        records.append(
            Record(
                sentence_id=f"{path.stem}:{block_number:06d}",
                block_number=block_number,
                source_file=path.name,
                original_tokenized=original_tokenized,
                original=original,
                gold_tokenized=" ".join(corrected_tokens),
                gold_correction=simple_detokenize(corrected_tokens),
                token_count=len(original_tokens),
                status=status,
                edit_count=len(real_edits),
                error_types=error_types,
                possible_context_dependency=int(bool(CONTEXT_HINT_RE.search(original))),
                possible_name_or_entity=int(bool(POSSIBLE_NAME_RE.search(original))),
                automatic_filter_reasons=";".join(sorted(set(reasons))),
            )
        )

    return records


# REPRODUCIBLE STRATIFIED SAMPLING

def stratified_sample(
    eligible: Sequence[Record], sample_size: int, correct_share: float, seed: int
) -> tuple[list[Record], dict[str, int]]:
    rng = random.Random(seed)
    correct = [record for record in eligible if record.status == "correct"]
    incorrect = [record for record in eligible if record.status == "incorrect"]

    desired_correct = round(sample_size * correct_share)
    desired_incorrect = sample_size - desired_correct

    chosen_correct = rng.sample(correct, min(desired_correct, len(correct)))
    chosen_incorrect = rng.sample(incorrect, min(desired_incorrect, len(incorrect)))
    chosen_ids = {record.sentence_id for record in chosen_correct + chosen_incorrect}

    remaining_needed = min(sample_size, len(eligible)) - len(chosen_ids)
    if remaining_needed:
        remaining = [record for record in eligible if record.sentence_id not in chosen_ids]
        extra = rng.sample(remaining, min(remaining_needed, len(remaining)))
    else:
        extra = []

    sample = chosen_correct + chosen_incorrect + extra
    rng.shuffle(sample)

    counts = {
        "sample_total": len(sample),
        "sample_correct": sum(record.status == "correct" for record in sample),
        "sample_incorrect": sum(record.status == "incorrect" for record in sample),
        "desired_correct": desired_correct,
        "desired_incorrect": desired_incorrect,
    }
    return sample, counts


# OUTPUT COLUMN DEFINITIONS

FULL_FIELDS = [
    "candidate_rank",
    "sentence_id",
    "block_number",
    "source_file",
    "original_tokenized",
    "original",
    "gold_tokenized",
    "gold_correction",
    "token_count",
    "status",
    "edit_count",
    "error_types",
    "possible_context_dependency",
    "possible_name_or_entity",
    "automatic_filter_reasons",
    "manual_intelligible",
    "manual_context_independent",
    "manual_pii_safe",
    "manual_category",
    "manual_exclude_reason",
    "manual_notes",
]

BLIND_FIELDS = [
    "review_rank",
    "sentence_id",
    "original",
    "token_count",
    "possible_context_dependency",
    "possible_name_or_entity",
    "manual_intelligible",
    "manual_context_independent",
    "manual_pii_safe",
    "manual_category",
    "manual_exclude_reason",
    "manual_notes",
]

EXCLUDED_FIELDS = [
    "sentence_id",
    "block_number",
    "source_file",
    "original",
    "token_count",
    "status",
    "automatic_filter_reasons",
]


# OUTPUT ROW PREPARATION

def record_to_candidate_row(record: Record, rank: int) -> dict[str, object]:
    row: dict[str, object] = asdict(record)
    row["candidate_rank"] = rank
    row.update(
        {
            "manual_intelligible": "",
            "manual_context_independent": "",
            "manual_pii_safe": "",
            "manual_category": "",
            "manual_exclude_reason": "",
            "manual_notes": "",
        }
    )
    return row


# CSV FILE WRITING

def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# MAIN DATA-PROCESSING PIPELINE

def main() -> int:
    args = parse_arguments()

    try:
        if not INPUT_FILE.is_file():
            raise FileNotFoundError(
                "Dataset not found. Place B.dev.gold.bea19.m2 at "
                f"{INPUT_FILE}"
            )
        records = parse_records(
            INPUT_FILE,
            annotator_id=str(args.annotator),
            minimum=args.min_tokens,
            maximum=args.max_tokens,
        )
    except (FileNotFoundError, ValueError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    eligible = [record for record in records if not record.automatic_filter_reasons]
    excluded = [record for record in records if record.automatic_filter_reasons]
    sample, sample_counts = stratified_sample(
        eligible,
        sample_size=args.sample_size,
        correct_share=args.correct_share,
        seed=args.seed,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT_DIR / "candidate_sentences.csv"
    blind_path = OUTPUT_DIR / "manual_review_blind.csv"
    excluded_path = OUTPUT_DIR / "excluded_sentences.csv"
    summary_path = OUTPUT_DIR / "selection_summary.json"

    candidate_rows = [record_to_candidate_row(record, rank) for rank, record in enumerate(sample, 1)]
    write_csv(full_path, FULL_FIELDS, candidate_rows)

    
    blind_rows = []
    blind_order = list(sample)
    random.Random(args.seed + 1).shuffle(blind_order)
    for rank, record in enumerate(blind_order, 1):
        blind_rows.append(
            {
                "review_rank": rank,
                "sentence_id": record.sentence_id,
                "original": record.original,
                "token_count": record.token_count,
                "possible_context_dependency": record.possible_context_dependency,
                "possible_name_or_entity": record.possible_name_or_entity,
                "manual_intelligible": "",
                "manual_context_independent": "",
                "manual_pii_safe": "",
                "manual_category": "",
                "manual_exclude_reason": "",
                "manual_notes": "",
            }
        )
    write_csv(blind_path, BLIND_FIELDS, blind_rows)

    excluded_rows = [asdict(record) for record in excluded]
    write_csv(excluded_path, EXCLUDED_FIELDS, excluded_rows)

    summary = {
        "input_file": str(INPUT_FILE),
        "annotator_id": str(args.annotator),
        "seed": args.seed,
        "minimum_tokens": args.min_tokens,
        "maximum_tokens": args.max_tokens,
        "requested_sample_size": args.sample_size,
        "requested_correct_share": args.correct_share,
        "parsed_blocks": len(records),
        "eligible_total": len(eligible),
        "eligible_correct": sum(record.status == "correct" for record in eligible),
        "eligible_incorrect": sum(record.status == "incorrect" for record in eligible),
        "automatically_excluded_total": len(excluded),
        **sample_counts,
       
        "outputs": {
            "candidate_sentences": str(full_path),
            "manual_review_blind": str(blind_path),
            "excluded_sentences": str(excluded_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input file:             {INPUT_FILE}")
    print(f"Parsed M2 blocks:       {len(records)}")
    print(f"Eligible sentences:     {len(eligible)}")
    print(f"Automatically excluded: {len(excluded)}")
    print(f"Sampled sentences:      {sample_counts['sample_total']}")
    print(f"  Incorrect:            {sample_counts['sample_incorrect']}")
    print(f"  Correct controls:     {sample_counts['sample_correct']}")
    print(f"\nCreated: {full_path}")
    print(f"Created: {blind_path}")
    print(f"Created: {excluded_path}")
    print(f"Created: {summary_path}")
   
    return 0


# SCRIPT ENTRY POINT

if __name__ == "__main__":
    raise SystemExit(main())
