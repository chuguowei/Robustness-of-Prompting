"""
rop/ape/instruction_gen.py

APE-based automatic instruction generation.

Two instruction types are generated (Section III-C and III-D of the paper):

  1. Error Correction Instruction (inec)
     Input:  perturbed question  →  Output: original question
     Used in: Stage 1 (Corrector)

  2. Guidance Instruction (inopt)
     Input:  corrected question  →  Output: ground-truth answer
     Used in: Stage 2 (Answerer)

Both use the APE library (Zhou et al., ICLR 2023) with a simple "forward"
prompt generation mode.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import ape
import data as ape_data
import config as ape_config

logger = logging.getLogger(__name__)


def _load_ape_config(
    eval_model: str,
    prompt_gen_model: str,
    num_prompts: int,
    eval_rounds: int,
    prompt_gen_batch_size: int,
    eval_batch_size: int,
) -> object:
    return ape_config.simple_config(
        eval_model=eval_model,
        prompt_gen_model=prompt_gen_model,
        prompt_gen_mode="forward",
        num_prompts=num_prompts,
        eval_rounds=eval_rounds,
        prompt_gen_batch_size=prompt_gen_batch_size,
        eval_batch_size=eval_batch_size,
    )


def _run_ape(inputs: list[str], outputs: list[str], conf: object):
    """Run APE and return (result, demo_fn)."""
    eval_template = "Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]"
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    prompt_gen_template = ape.get_simple_prompt_gen_template(None, "forward")

    split_size = min(int(len(inputs) * 0.5), 100)
    prompt_gen_data, eval_data = ape_data.create_split(
        (inputs, outputs), split_size
    )

    return ape.find_prompts(
        eval_template=eval_template,
        demos_template=demos_template,
        prompt_gen_data=prompt_gen_data,
        eval_data=eval_data,
        conf=conf,
        prompt_gen_template=prompt_gen_template,
    )


class InstructionGenerator:
    """
    Generates error correction and guidance instructions via APE.

    Usage::

        gen = InstructionGenerator(cfg)

        # Error correction instruction
        inec = gen.generate_correction_instruction("results/aqua/EC/perturbed.json")

        # Guidance instruction
        inopt = gen.generate_guidance_instruction("results/aqua/EC/corrected.json")
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: The 'ape' section of configs/default.yaml, e.g.
                 {"eval_model": "gpt-3.5-turbo-1106",
                  "prompt_gen_model": "gpt-3.5-turbo-1106",
                  "num_prompts": 50, ...}
        """
        self.cfg = cfg

    def _make_conf(self):
        return _load_ape_config(
            eval_model=self.cfg.get("eval_model", "gpt-3.5-turbo-1106"),
            prompt_gen_model=self.cfg.get("prompt_gen_model", "gpt-3.5-turbo-1106"),
            num_prompts=self.cfg.get("num_prompts", 50),
            eval_rounds=self.cfg.get("eval_rounds", 20),
            prompt_gen_batch_size=self.cfg.get("prompt_gen_batch_size", 200),
            eval_batch_size=self.cfg.get("eval_batch_size", 500),
        )

    def generate_correction_instruction(self, perturbed_file: str) -> Optional[str]:
        """
        Generate inec from perturbed↔original question pairs.

        Expected JSON schema (list of objects):
          [{"rewritten_question": "...", "original_question": "..."}, ...]

        Returns the best instruction string, or None on failure.
        """
        with open(perturbed_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        inputs = [item["rewritten_question"] for item in data]
        outputs = [item["original_question"] for item in data]

        logger.info(
            "Generating error correction instruction from %d samples…", len(inputs)
        )
        result, _ = _run_ape(inputs, outputs, self._make_conf())
        prompts, scores = result.sorted()
        best = prompts[0]
        logger.info("Best correction instruction (score=%.4f): %s", scores[0], best)
        return best

    def generate_guidance_instruction(self, corrected_file: str) -> Optional[str]:
        """
        Generate inopt from corrected question↔answer pairs.

        Expected JSON schema (list of objects):
          [{"response_1": "...", "original_answer": "..."}, ...]

        Returns the best instruction string, or None on failure.
        """
        with open(corrected_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        inputs = [item["response_1"] for item in data]
        outputs = [item["original_answer"] for item in data]

        logger.info(
            "Generating guidance instruction from %d samples…", len(inputs)
        )
        result, _ = _run_ape(inputs, outputs, self._make_conf())
        prompts, scores = result.sorted()
        best = prompts[0]
        logger.info("Best guidance instruction (score=%.4f): %s", scores[0], best)
        return best
