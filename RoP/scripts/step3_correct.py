#!/usr/bin/env python3
"""
scripts/step3_correct.py

Step 3 — Error Correction (Stage 1 of RoP).

Reads perturbed.json and correction_instruction.txt, applies the correction
instruction to each perturbed question, and writes corrected.json.

Usage:
    python scripts/step3_correct.py --data_dir results/aqua/EC

Output file:  <data_dir>/corrected.json
Schema:
    [{"rewritten_question":  "<instruction> + <perturbed>",
      "response_1":          "<corrected question>",
      "original_answer":     "..."},
     ...]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rop.llm.client import LLMClient
from rop.correction.corrector import Corrector
from rop.utils.data import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3: Error Correction")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir)

    perturbed_file = data_dir / "perturbed.json"
    instruction_file = data_dir / "correction_instruction.txt"

    for p in (perturbed_file, instruction_file):
        if not p.exists():
            logger.error("Required file missing: %s", p)
            sys.exit(1)

    data = load_json(str(perturbed_file))
    instruction = instruction_file.read_text(encoding="utf-8").strip()

    llm_cfg = cfg["llm"]
    client = LLMClient(
        model=llm_cfg["optimizer_model"],
        temperature=llm_cfg["temperature"],
        max_retries=llm_cfg["max_retries"],
        retry_interval=llm_cfg["retry_interval"],
        request_interval=llm_cfg["request_interval"],
    )
    corrector = Corrector(client, instruction)

    results = []
    for i, item in enumerate(data):
        logger.info("[%d/%d] Correcting…", i + 1, len(data))
        corrected = corrector.correct(
            perturbed_question=item["rewritten_question"],
            original_question=item["original_question"],
        )
        results.append(
            {
                "rewritten_question": instruction + "\n\n" + item["rewritten_question"],
                "response_1": corrected,
                "original_answer": item["original_answer"],
            }
        )

    save_json(results, str(data_dir / "corrected.json"))
    logger.info("Step 3 complete → %s", data_dir / "corrected.json")


if __name__ == "__main__":
    main()
