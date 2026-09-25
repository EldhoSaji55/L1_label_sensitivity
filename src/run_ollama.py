from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


INPUT_FIELDS = ["S_No", "sentence_id", "language", "Sentence"]
OUTPUT_FIELDS = INPUT_FIELDS + ["model_answer", "model", "response_id", "finish_reason"]
SYSTEM_PROMPT = (
    "You correct English learner sentences. Make only necessary corrections, "
    "preserve the intended meaning, and return only the corrected English sentence. "
    "If the sentence is already correct, return it unchanged."
)


def read_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INPUT_FIELDS:
            raise ValueError(f"Expected exactly these columns: {INPUT_FIELDS}")
        rows = list(reader)
    if not rows or any(not all(row[field].strip() for field in INPUT_FIELDS) for row in rows):
        raise ValueError("The job file is empty or has blank required cells.")
    if len({row["S_No"] for row in rows}) != len(rows):
        raise ValueError("S_No values must be unique.")

    labels = {row["language"] for row in rows}
    if len(labels) < 2:
        raise ValueError("The job file must contain at least two language conditions.")
    if "Unknown" not in labels:
        raise ValueError("The job file must include Unknown as the no-L1 baseline.")

    counts = Counter(row["sentence_id"] for row in rows)
    if any(count != len(labels) for count in counts.values()):
        raise ValueError(
            f"Every sentence_id must occur exactly {len(labels)} times."
        )
    for sentence_id in counts:
        grouped = [row for row in rows if row["sentence_id"] == sentence_id]
        if {row["language"] for row in grouped} != labels:
            raise ValueError(f"Missing label condition for {sentence_id}")
        if len({row["Sentence"] for row in grouped}) != 1:
            raise ValueError(f"Sentence text changes across labels for {sentence_id}")
    return rows


def user_prompt(row: dict[str, str]) -> str:
    if row["language"] == "Unknown":
        label = "No information about the learner's first language is available."
    else:
        label = f"The learner's first language is {row['language']}."
    return f"{label}\nEnglish learner sentence: {row['Sentence']}"


def request_model(
    row: dict[str, str], *, model: str, host: str, timeout: int, num_predict: int
) -> tuple[str, str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
        ],
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "keep_alive": "10m",
        "stream": False,
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            answer = result.get("message", {}).get("content", "")
            if not isinstance(answer, str) or not answer.strip():
                reason = result.get("done_reason", "unknown")
                raise RuntimeError(f"The model returned no final text (done_reason={reason}).")
            return answer.strip(), result.get("created_at", ""), result.get("done_reason", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Connection error: {exc}") from exc
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("Unreachable")


def completed_numbers(
    output: Path, source_by_number: dict[str, dict[str, str]], model: str
) -> set[str]:
    if not output.exists():
        return set()
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError(f"Unexpected columns in existing output: {output}")
        completed: set[str] = set()
        for result in reader:
            number = result["S_No"]
            source = source_by_number.get(number)
            if source is None or any(result[field] != source[field] for field in INPUT_FIELDS):
                raise ValueError(f"Input changed since result S_No {number} was saved.")
            if result["model"] != model:
                raise ValueError("Existing output uses another model; choose a different file.")
            if number in completed:
                raise ValueError(f"Duplicate result S_No: {number}")
            completed.add(number)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Installed Ollama tag, e.g. qwen3.5:4b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pending jobs; otherwise run one sentence across all labels",
    )
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ.")
    jobs = read_jobs(args.input)
    first_id = jobs[0]["sentence_id"]
    target = jobs if args.all else [row for row in jobs if row["sentence_id"] == first_id]
    done = completed_numbers(args.output, {row["S_No"]: row for row in jobs}, args.model)
    pending = [row for row in target if row["S_No"] not in done]
    if not pending:
        print("Nothing left to run in this selection.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.output.exists()
    with args.output.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if is_new:
            writer.writeheader()
            handle.flush()
        for index, row in enumerate(pending, 1):
            try:
                answer, response_id, finish_reason = request_model(
                    row,
                    model=args.model,
                    host=args.host,
                    timeout=args.timeout,
                    num_predict=args.num_predict,
                )
            except Exception as exc:  # preserve completed rows before stopping
                print(f"Stopped at S_No {row['S_No']}: {exc}", file=sys.stderr)
                print("Completed rows are saved. Check Ollama and rerun the same command.", file=sys.stderr)
                return 1
            writer.writerow(
                {
                    **row,
                    "model_answer": answer,
                    "model": args.model,
                    "response_id": response_id,
                    "finish_reason": finish_reason,
                }
            )
            handle.flush()
            print(f"Saved {index}/{len(pending)}: S_No {row['S_No']} ({row['language']})")
    print(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())