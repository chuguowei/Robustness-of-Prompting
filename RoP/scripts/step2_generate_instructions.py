#!/usr/bin/env python3
"""
scripts/step2_generate_instructions.py

Step 2 — Automatic Prompt Engineering (APE).

Reads perturbed.json and generates two instructions via APE:
  - correction_instruction.txt  (inec  — for Stage 1)
  - guidance_instruction.txt    (inopt — for Stage 2)

Usage:
    python scripts/step2_generate_instructions.py --data_dir results/aqua/EC

Both files are saved into <data_dir>/.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rop.ape.instruction_gen import InstructionGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: Generate APE Instructions")
    parser.add_argument("--data_dir", required=True, help="Directory containing perturbed.json")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    perturbed_file = data_dir / "perturbed.json"

    if not perturbed_file.exists():
        logger.error("perturbed.json not found in %s. Run step1 first.", data_dir)
        sys.exit(1)

    gen = InstructionGenerator(cfg["ape"])

    # --- Error Correction Instruction ---
    logger.info("=== Generating Error Correction Instruction ===")
    inec = gen.generate_correction_instruction(str(perturbed_file))
    if inec:
        out = data_dir / "correction_instruction.txt"
        out.write_text(inec, encoding="utf-8")
        logger.info("Saved correction instruction → %s", out)
    else:
        logger.error("Failed to generate correction instruction.")

    # --- Guidance Instruction ---
    # At this point corrected.json does not exist yet; we use perturbed.json
    # with response_1 ≡ original_question as a proxy — or the user can re-run
    # this script after step3 to regenerate inopt from corrected data.
    corrected_file = data_dir / "corrected.json"
    if corrected_file.exists():
        logger.info("=== Generating Guidance Instruction (from corrected.json) ===")
        inopt = gen.generate_guidance_instruction(str(corrected_file))
    else:
        logger.warning(
            "corrected.json not found — generating guidance instruction from "
            "perturbed.json (rerun after step3 for best results)."
        )
        inopt = gen.generate_guidance_instruction(str(perturbed_file))

    if inopt:
        out = data_dir / "guidance_instruction.txt"
        out.write_text(inopt, encoding="utf-8")
        logger.info("Saved guidance instruction → %s", out)
    else:
        logger.error("Failed to generate guidance instruction.")

    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()
