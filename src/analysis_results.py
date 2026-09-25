from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    return re.sub(r"\s+", " ", text)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def language_key(label: str) -> str:
    """Convert a language label into a safe output-column suffix."""
    key = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not key:
        raise ValueError(f"Cannot create a column name for language label: {label!r}")
    return key


def validate_results(results: pd.DataFrame) -> tuple[str, ...]:
    required = {"S_No", "sentence_id", "language", "Sentence", "model_answer", "model"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if results["S_No"].duplicated().any():
        raise ValueError("S_No values are not unique.")
    if results[["sentence_id", "language", "Sentence", "model_answer", "model"]].isna().any().any():
        raise ValueError("Required result cells cannot be empty.")
    if results[["sentence_id", "language", "Sentence", "model_answer", "model"]].apply(
        lambda column: column.astype(str).str.strip().eq("")
    ).any().any():
        raise ValueError("Required result cells cannot be blank.")
    if results["model"].nunique() != 1:
        raise ValueError("One result file must contain exactly one model.")

    labels = tuple(
        sorted(
            results["language"].astype(str).str.strip().unique(),
            key=lambda label: (label.casefold() == "unknown", label.casefold()),
        )
    )
    if len(labels) < 2:
        raise ValueError("Results must contain at least two language conditions.")
    if "Unknown" not in labels:
        raise ValueError("Results must include Unknown as the no-L1 baseline.")

    keys = [language_key(label) for label in labels]
    if len(keys) != len(set(keys)):
        raise ValueError("Two language labels produce the same output-column name.")

    counts = results.groupby("sentence_id")["language"].agg(list)
    expected = set(labels)
    for sentence_id, sentence_labels in counts.items():
        if len(sentence_labels) != len(labels) or set(sentence_labels) != expected:
            raise ValueError(
                f"{sentence_id} does not contain exactly the complete label set."
            )
    sentence_variants = results.groupby("sentence_id")["Sentence"].nunique()
    if (sentence_variants != 1).any():
        raise ValueError("Sentence text changes across label conditions.")
    return labels


def analyse(results: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = validate_results(results)
    sentence_col = "original" if "original" in reference.columns else "Sentence"
    required_ref = {"sentence_id", sentence_col, "recommended_category", "gold_correction"}
    missing = required_ref.difference(reference.columns)
    if missing:
        raise ValueError(f"Missing reference columns: {sorted(missing)}")
    if reference["sentence_id"].duplicated().any():
        raise ValueError("Reference sentence_id values must be unique.")

    ref = reference.copy()
    ref = ref.rename(columns={sentence_col: "reference_sentence"})
    if "gold_status" not in ref:
        ref["gold_status"] = ref.apply(
            lambda row: "correct"
            if normalize_text(row["reference_sentence"]) == normalize_text(row["gold_correction"])
            else "incorrect",
            axis=1,
        )
    keep = [
        "sentence_id",
        "reference_sentence",
        "recommended_category",
        "gold_status",
        "gold_correction",
    ]
    label_level = results.merge(ref[keep], on="sentence_id", how="left", validate="many_to_one")
    if label_level["recommended_category"].isna().any():
        raise ValueError("Some result sentence IDs are absent from the reference.")
    mismatched_input = label_level.apply(
        lambda row: normalize_text(row["Sentence"]) != normalize_text(row["reference_sentence"]),
        axis=1,
    )
    if mismatched_input.any():
        raise ValueError("At least one result sentence differs from the verified reference.")

    label_level["normalized_input"] = label_level["Sentence"].map(normalize_text)
    label_level["normalized_answer"] = label_level["model_answer"].map(normalize_text)
    label_level["normalized_gold"] = label_level["gold_correction"].map(normalize_text)
    label_level["changed_from_input"] = label_level["normalized_answer"] != label_level["normalized_input"]
    label_level["exact_gold_match"] = label_level["normalized_answer"] == label_level["normalized_gold"]

    records: list[dict[str, object]] = []
    for sentence_id, group in label_level.groupby("sentence_id", sort=False):
        first = group.iloc[0]
        by_label = group.set_index("language")
        record: dict[str, object] = {
            "sentence_id": sentence_id,
            "Sentence": first["Sentence"],
            "recommended_category": first["recommended_category"],
            "gold_status": first["gold_status"],
            "gold_correction": first["gold_correction"],
            "number_of_labels": len(labels),
            "labels_tested": ";".join(labels),
        }
        normalized: list[str] = []
        matches: list[bool] = []
        for label in labels:
            key = language_key(label)
            row = by_label.loc[label]
            record[f"answer_{key}"] = row["model_answer"]
            record[f"normalized_{key}"] = row["normalized_answer"]
            record[f"gold_match_{key}"] = bool(row["exact_gold_match"])
            normalized.append(row["normalized_answer"])
            matches.append(bool(row["exact_gold_match"]))
        unique_answers = len(set(normalized))
        sensitive = unique_answers > 1
        record.update(
            {
                "unique_answers": unique_answers,
                "label_sensitive": yes_no(sensitive),
                "number_matching_gold": sum(matches),
                "gold_match_varies_by_label": yes_no(len(set(matches)) > 1),
                "difference_type": "",
                "quality_difference": "",
                "reviewer_notes": "",
                "needs_manual_review": yes_no(sensitive),
            }
        )
        records.append(record)

    sentence_level = pd.DataFrame(records)
    summary = (
        sentence_level.assign(sensitive=sentence_level["label_sensitive"].eq("Yes").astype(int))
        .groupby("recommended_category", sort=False)
        .agg(total=("sentence_id", "size"), sensitive=("sensitive", "sum"))
        .reset_index()
    )
    summary["sensitivity_rate_percent"] = (summary["sensitive"] / summary["total"] * 100).round(2)
    return label_level, sentence_level, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results, encoding="utf-8-sig")
    reference = pd.read_csv(args.reference, encoding="utf-8-sig")
    label_level, sentence_level, summary = analyse(results, reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_level.to_csv(args.output_dir / "label_level_analysis.csv", index=False, encoding="utf-8-sig")
    sentence_level.to_csv(args.output_dir / "sentence_level_analysis.csv", index=False, encoding="utf-8-sig")
    sentence_level[sentence_level["label_sensitive"].eq("Yes")].to_csv(
        args.output_dir / "sensitive_cases_for_manual_review.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(args.output_dir / "category_summary.csv", index=False, encoding="utf-8-sig")

    sensitive = int(sentence_level["label_sensitive"].eq("Yes").sum())
    total = len(sentence_level)
    print(f"Validated {len(label_level)} outputs for {total} unique sentences.")
    print(f"Label-sensitive: {sensitive}/{total} ({sensitive / total * 100:.2f}%)")
    print(f"Created analysis files in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())