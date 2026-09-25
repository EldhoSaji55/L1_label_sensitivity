from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


KEY_COLUMNS = ["sentence_id", "language"]
SHARED_COLUMNS = ["Sentence", "recommended_category", "gold_correction"]
REQUIRED_COLUMNS = {
    "S_No",
    *KEY_COLUMNS,
    *SHARED_COLUMNS,
    "model_answer",
    "exact_gold_match",
}

CATEGORY_ORDER = [
    "both_models_error",
    "qwen_only_error",
    "gemma_only_error",
    "both_models_match",
]

CATEGORY_DESCRIPTIONS = {
    "both_models_error": "Neither model exactly matches the gold correction",
    "qwen_only_error": "Only Qwen differs from the gold correction",
    "gemma_only_error": "Only Gemma differs from the gold correction",
    "both_models_match": "Both models exactly match the gold correction",
}


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Convert common CSV truth values into a strict Boolean series."""
    normalized = series.astype(str).str.strip().str.casefold()
    valid = normalized.isin({"true", "false", "yes", "no", "1", "0"})
    if not valid.all():
        invalid = sorted(normalized[~valid].unique())
        raise ValueError(f"Invalid Boolean values in {column_name}: {invalid}")
    return normalized.isin({"true", "yes", "1"})


def validate_input(frame: pd.DataFrame, model_name: str) -> None:
    """Validate one label-level analysis file before comparison."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{model_name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{model_name} contains no rows.")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{model_name} contains empty comparison keys.")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            f"{model_name} contains duplicate sentence_id/language combinations."
        )


def comparison_category(qwen_non_match: bool, gemma_non_match: bool) -> str:
    if qwen_non_match and gemma_non_match:
        return "both_models_error"
    if qwen_non_match:
        return "qwen_only_error"
    if gemma_non_match:
        return "gemma_only_error"
    return "both_models_match"


def compare(qwen: pd.DataFrame, gemma: pd.DataFrame) -> pd.DataFrame:
    """Merge matching sentence-label conditions and assign four-way outcomes."""
    validate_input(qwen, "Qwen")
    validate_input(gemma, "Gemma")

    keep = [
        "S_No",
        *KEY_COLUMNS,
        *SHARED_COLUMNS,
        "model_answer",
        "exact_gold_match",
    ]
    merged = qwen[keep].merge(
        gemma[keep],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_qwen", "_gemma"),
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, KEY_COLUMNS + ["_merge"]].head(10)
        raise ValueError(
            "The model files do not contain identical sentence-language rows. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    for column in SHARED_COLUMNS:
        left = merged[f"{column}_qwen"].astype(str)
        right = merged[f"{column}_gemma"].astype(str)
        if not left.equals(right):
            raise ValueError(f"Qwen and Gemma differ in shared column: {column}")

    merged["qwen_exact_gold_match"] = parse_boolean(
        merged["exact_gold_match_qwen"], "Qwen exact_gold_match"
    )
    merged["gemma_exact_gold_match"] = parse_boolean(
        merged["exact_gold_match_gemma"], "Gemma exact_gold_match"
    )
    merged["qwen_non_gold_match"] = ~merged["qwen_exact_gold_match"]
    merged["gemma_non_gold_match"] = ~merged["gemma_exact_gold_match"]
    merged["comparison_category"] = [
        comparison_category(qwen_error, gemma_error)
        for qwen_error, gemma_error in zip(
            merged["qwen_non_gold_match"], merged["gemma_non_gold_match"]
        )
    ]
    merged["category_description"] = merged["comparison_category"].map(
        CATEGORY_DESCRIPTIONS
    )

    detailed = pd.DataFrame(
        {
            "S_No": merged["S_No_qwen"],
            "sentence_id": merged["sentence_id"],
            "language": merged["language"],
            "Sentence": merged["Sentence_qwen"],
            "recommended_category": merged["recommended_category_qwen"],
            "gold_correction": merged["gold_correction_qwen"],
            "qwen_answer": merged["model_answer_qwen"],
            "gemma_answer": merged["model_answer_gemma"],
            "qwen_exact_gold_match": merged["qwen_exact_gold_match"],
            "gemma_exact_gold_match": merged["gemma_exact_gold_match"],
            "qwen_non_gold_match": merged["qwen_non_gold_match"],
            "gemma_non_gold_match": merged["gemma_non_gold_match"],
            "comparison_category": merged["comparison_category"],
            "category_description": merged["category_description"],
            "manual_qwen_correct": "",
            "manual_gemma_correct": "",
            "manual_final_category": "",
            "reviewer_notes": "",
        }
    )
    return detailed.sort_values(["sentence_id", "language"]).reset_index(drop=True)


def four_way_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    total = len(detailed)
    counts = detailed["comparison_category"].value_counts()
    return pd.DataFrame(
        [
            {
                "comparison_category": category,
                "description": CATEGORY_DESCRIPTIONS[category],
                "count": int(counts.get(category, 0)),
                "percent": round(int(counts.get(category, 0)) / total * 100, 2),
            }
            for category in CATEGORY_ORDER
        ]
    )


def grouped_summary(detailed: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for value, group in detailed.groupby(group_column, sort=False):
        counts = group["comparison_category"].value_counts()
        records.append(
            {
                group_column: value,
                "total_conditions": len(group),
                "both_models_error": int(
                    counts.get("both_models_error", 0)
                ),
                "qwen_only_error": int(
                    counts.get("qwen_only_error", 0)
                ),
                "gemma_only_error": int(
                    counts.get("gemma_only_error", 0)
                ),
                "both_models_match": int(
                    counts.get("both_models_match", 0)
                ),
                "qwen_total_non_gold_match": int(group["qwen_non_gold_match"].sum()),
                "gemma_total_non_gold_match": int(group["gemma_non_gold_match"].sum()),
            }
        )
    return pd.DataFrame(records)


def model_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    total = len(detailed)
    return pd.DataFrame(
        [
            {
                "model": model,
                "total_conditions": total,
                "exact_gold_matches": int(detailed[f"{key}_exact_gold_match"].sum()),
                "non_gold_matches": int(detailed[f"{key}_non_gold_match"].sum()),
                "non_gold_match_rate_percent": round(
                    detailed[f"{key}_non_gold_match"].mean() * 100, 2
                ),
            }
            for model, key in (("Qwen", "qwen"), ("Gemma", "gemma"))
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qwen = pd.read_csv(args.qwen, encoding="utf-8-sig")
    gemma = pd.read_csv(args.gemma, encoding="utf-8-sig")
    detailed = compare(qwen, gemma)
    summary = four_way_summary(detailed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(
        args.output_dir / "error_comparison_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "error_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_summary(detailed).to_csv(
        args.output_dir / "model_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped_summary(detailed, "language").to_csv(
        args.output_dir / "error_comparison_by_language.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped_summary(detailed, "recommended_category").to_csv(
        args.output_dir / "error_comparison_by_sentence_category.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts = dict(zip(summary["comparison_category"], summary["count"]))
    print(f"Compared sentence-language conditions: {len(detailed)}")
    print(
        "Both models non-gold match: "
        f"{counts.get('both_models_error', 0)}"
    )
    print(f"Qwen-only non-gold match: {counts.get('qwen_only_error', 0)}")
    print(f"Gemma-only non-gold match: {counts.get('gemma_only_error', 0)}")
    print(
        "Both models exact-gold match: "
        f"{counts.get('both_models_match', 0)}"
    )
    print(f"Created comparison files in: {args.output_dir}")
 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
