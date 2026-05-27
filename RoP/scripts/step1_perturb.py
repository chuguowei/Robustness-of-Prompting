#!/usr/bin/env python3
"""
scripts/step1_perturb.py

Step 1 — Question Perturbation.

Reads a clean dataset, applies the chosen perturbation strategy to each
question using GPT-4o, and saves the adversarial examples.

Usage:
    python scripts/step1_perturb.py \\
        --dataset aqua \\
        --perturb_type EC \\
        --output_dir results/aqua/EC

Output file:  <output_dir>/perturbed.json
Schema:
    [{"original_question": "...",
      "original_answer":   "...",
      "rewritten_question": "..."},
     ...]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rop.llm.client import LLMClient
from rop.perturbation.perturber import Perturber
from rop.utils.data import load_dataset, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Question Perturbation")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. aqua, gsm8k)")
    parser.add_argument(
        "--perturb_type",
        required=True,
        choices=["EC", "SC", "WOO", "HW", "UIC"],
        help="Perturbation strategy",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--datasets_config", default="configs/datasets.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.datasets_config) as f:
        ds_cfg = yaml.safe_load(f)

    # --- Load dataset ---
    ds_info = ds_cfg["datasets"].get(args.dataset)
    if ds_info is None:
        logger.error("Unknown dataset '%s'. Check configs/datasets.yaml.", args.dataset)
        sys.exit(1)

    max_samples = args.max_samples or cfg["pipeline"].get("max_samples")
    records = load_dataset(args.dataset, ds_info["path"], max_samples)

    # --- Set up LLM client & perturber ---
    llm_cfg = cfg["llm"]
    client = LLMClient(
        model=llm_cfg["optimizer_model"],
        temperature=llm_cfg["temperature"],
        max_retries=llm_cfg["max_retries"],
        retry_interval=llm_cfg["retry_interval"],
        request_interval=llm_cfg["request_interval"],
    )
    perturber = Perturber(client, args.perturb_type)

    # --- Perturb ---
    results = []
    for i, item in enumerate(records):
        logger.info("[%d/%d] Perturbing…", i + 1, len(records))
        perturbed = perturber.perturb(item["question"], item["answer"])
        results.append(
            {
                "original_question": item["question"],
                "original_answer": item["answer"],
                "rewritten_question": perturbed or item["question"],
            }
        )

    # --- Save ---
    output_path = str(Path(args.output_dir) / "perturbed.json")
    save_json(results, output_path)
    logger.info("Done. Saved %d examples → %s", len(results), output_path)


if __name__ == "__main__":
    main()
