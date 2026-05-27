#!/usr/bin/env python3
"""
scripts/step4_answer.py

Step 4 — Guided Answering (Stage 2 of RoP).

Reads corrected.json and guidance_instruction.txt, asks the evaluator model
(GPT-3.5-Turbo) to answer each corrected question, extracts the predicted
answer, and computes overall accuracy.

Usage:
    python scripts/step4_answer.py \\
        --data_dir results/aqua/EC \\
        --answer_type multiple_choice

Output file:  <data_dir>/final_answers.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rop.llm.client import LLMClient
from rop.guidance.answerer import Answerer
from rop.utils.data import load_json, save_json
from rop.utils.metrics import print_accuracy_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: Guided Answering")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--answer_type",
        default="multiple_choice",
        choices=["multiple_choice", "numeric"],
        help="How to extract the answer from the model's response",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    corrected_file = data_dir / "corrected.json"
    instruction_file = data_dir / "guidance_instruction.txt"

    for p in (corrected_file, instruction_file):
        if not p.exists():
            logger.error("Required file missing: %s", p)
            sys.exit(1)

    data = load_json(str(corrected_file))
    instruction = instruction_file.read_text(encoding="utf-8").strip()

    llm_cfg = cfg["llm"]
    client = LLMClient(
        model=llm_cfg["evaluator_model"],
        temperature=llm_cfg["temperature"],
        max_retries=llm_cfg["max_retries"],
        retry_interval=llm_cfg["retry_interval"],
        request_interval=llm_cfg["request_interval"],
    )
    answerer = Answerer(client, instruction, answer_type=args.answer_type)

    predictions, ground_truths, results = [], [], []

    for i, item in enumerate(data):
        logger.info("[%d/%d] Answering…", i + 1, len(data))
        corrected_q = item["response_1"]
        gold = item["original_answer"].replace(",", "")

        pred, raw = answerer.answer(corrected_q)

        predictions.append(pred)
        ground_truths.append(gold)

        results.append(
            {
                "input_question": instruction + " " + corrected_q,
                "response": raw,
                "predicted_answer": pred,
                "original_answer": gold,
            }
        )

    # Accuracy report
    print_accuracy_report(predictions, ground_truths, label=str(data_dir))

    save_json(results, str(data_dir / "final_answers.json"))
    logger.info("Step 4 complete → %s", data_dir / "final_answers.json")


if __name__ == "__main__":
    main()
