"""
rop/utils/data.py

Dataset loading and JSON I/O helpers.
"""

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic I/O
# ---------------------------------------------------------------------------

def load_json(path: str) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: list | dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info("Saved %d records → %s", len(data) if isinstance(data, list) else 1, path)


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _iter_aqua(path: str) -> Iterator[tuple[str, str]]:
    for item in load_json(path):
        q = item.get("question", "")
        choices = "  ".join(item.get("options", []))
        full_q = f"{q}  Answer Choices: {choices}"
        yield full_q, item.get("correct", "")


def _iter_gsm8k(path: str) -> Iterator[tuple[str, str]]:
    for item in load_jsonl(path):
        answer_text = item.get("answer", "")
        # GSM8K answers end with "#### <number>"
        numeric = answer_text.split("####")[-1].strip().replace(",", "")
        yield item.get("question", ""), numeric


def _iter_generic_json(path: str) -> Iterator[tuple[str, str]]:
    """Fallback for AddSub / MultiArith / SVAMP / SingleEq."""
    for item in load_json(path):
        q = item.get("sQuestion", item.get("question", ""))
        a = str(item.get("lSolutions", [item.get("answer", "")])[0])
        yield q, a


def _iter_commonsensqa(path: str) -> Iterator[tuple[str, str]]:
    for item in load_jsonl(path):
        stem = item["question"]["stem"]
        choices_text = "  ".join(
            f"({c['label']}) {c['text']}" for c in item["question"]["choices"]
        )
        q = f"{stem}  Answer Choices: {choices_text}"
        yield q, item.get("answerKey", "")


def _iter_bigbench(path: str) -> Iterator[tuple[str, str]]:
    data = load_json(path)
    for ex in data.get("examples", []):
        yield ex.get("input", ""), str(ex.get("target_scores", {}).get("True", ex.get("output", "")))


_LOADERS = {
    "aqua": _iter_aqua,
    "gsm8k": _iter_gsm8k,
    "singleeq": _iter_generic_json,
    "addsub": _iter_generic_json,
    "multiarith": _iter_generic_json,
    "svamp": _iter_generic_json,
    "commonsensqa": _iter_commonsensqa,
    "strategyqa": _iter_bigbench,
    "date_understanding": _iter_bigbench,
    "object_tracking": _iter_bigbench,
}


def load_dataset(dataset_name: str, path: str, max_samples: int | None = None) -> list[dict]:
    """
    Load a dataset into a unified list of {"question": ..., "answer": ...} dicts.

    Args:
        dataset_name: Key matching configs/datasets.yaml (e.g. "aqua").
        path:         Path to the dataset file.
        max_samples:  Truncate to this many samples if set.

    Returns:
        List of {"question": str, "answer": str} dicts.
    """
    loader = _LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"No loader for dataset '{dataset_name}'. "
            f"Available: {list(_LOADERS)}"
        )

    records = [
        {"question": q, "answer": a}
        for q, a in loader(path)
    ]

    if max_samples is not None:
        records = records[:max_samples]

    logger.info("Loaded %d samples from '%s'.", len(records), dataset_name)
    return records
