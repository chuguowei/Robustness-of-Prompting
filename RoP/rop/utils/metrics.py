"""
rop/utils/metrics.py

Accuracy evaluation helpers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def accuracy(
    predictions: list[Optional[str]],
    answers: list[str],
    ignore_empty: bool = True,
) -> float:
    """
    Compute exact-match accuracy.

    Args:
        predictions:  Model predicted answers (may contain None for failures).
        answers:      Ground-truth answers.
        ignore_empty: If True, skip pairs where prediction is None or "".

    Returns:
        Accuracy as a float in [0, 1].
    """
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length.")

    correct = 0
    total = 0

    for pred, ans in zip(predictions, answers):
        if ignore_empty and not pred:
            continue
        total += 1
        if pred == ans:
            correct += 1

    if total == 0:
        logger.warning("No valid predictions to evaluate.")
        return 0.0

    return correct / total


def print_accuracy_report(
    predictions: list[Optional[str]],
    answers: list[str],
    label: str = "",
) -> float:
    acc = accuracy(predictions, answers)
    tag = f"[{label}] " if label else ""
    print(f"{tag}Accuracy: {acc * 100:.2f}%  ({sum(p == a for p, a in zip(predictions, answers))} / {len(answers)})")
    return acc
