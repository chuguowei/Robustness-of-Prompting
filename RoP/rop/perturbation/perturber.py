"""
rop/perturbation/perturber.py

Step 1 — Question Perturbation.

Applies one of five perturbation strategies to generate adversarial examples
from original (clean) question-answer pairs, using an LLM as the perturbation
model (as described in Section III-B of the paper).
"""

import logging
from typing import Optional

from rop.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perturbation type descriptions (fed verbatim into the LLM prompt)
# ---------------------------------------------------------------------------
PERTURBATION_DESCRIPTIONS: dict[str, str] = {
    "EC": (
        "Choose some words in the sentence, and randomly shuffle their internal "
        "characters or change characters to others (e.g., times → tmies, will → wlil)."
    ),
    "SC": (
        "Replace one or more characters with visually similar symbols or special "
        "characters (e.g., will → wil̈l, times → tīmês)."
    ),
    "WOO": (
        "Rearrange neighboring word positions to disrupt syntactic structure "
        "(e.g., '6 times older' → 'older 6 times', '3 times' → 'times 3')."
    ),
    "HW": (
        "Replace original terms with phonetically equivalent alternatives "
        "(e.g., be → bee, eight → ate)."
    ),
    "UIC": (
        "Append irrelevant but plausible information that does not alter the answer "
        "(e.g., add an unrelated sentence about a past purchase or future plan)."
    ),
}

# ---------------------------------------------------------------------------
# Prompt template (mirrors the paper's Section III-B verbatim)
# ---------------------------------------------------------------------------
_PERTURBATION_PROMPT_TEMPLATE = """\
Your objective is to rewrite a given math question using the following perturbation strategy. \
The rewritten question should be reasonable, understandable, and able to be responded to by humans.

Perturbation strategy: {strategy}

The given question: {question}
Answer of the given question: {answer}

Please rewrite the question using the specified perturbation strategy, and avoid significant \
deviation in the question content.
It is important to ensure that the rewritten question has only one required numerical answer. \
You just need to print the rewritten question without answer."""


class Perturber:
    """Generates adversarial examples using a chosen perturbation type."""

    def __init__(self, client: LLMClient, perturb_type: str):
        if perturb_type not in PERTURBATION_DESCRIPTIONS:
            raise ValueError(
                f"Unknown perturbation type '{perturb_type}'. "
                f"Choose from: {list(PERTURBATION_DESCRIPTIONS)}"
            )
        self.client = client
        self.perturb_type = perturb_type
        self.strategy_description = PERTURBATION_DESCRIPTIONS[perturb_type]

    def perturb(self, question: str, answer: str) -> Optional[str]:
        """
        Generate one perturbed version of *question*.

        Args:
            question: Original clean question text.
            answer:   Ground-truth answer (helps the LLM preserve semantics).

        Returns:
            Perturbed question string, or None if the LLM call failed.
        """
        prompt = _PERTURBATION_PROMPT_TEMPLATE.format(
            strategy=self.strategy_description,
            question=question,
            answer=answer,
        )
        result = self.client.complete(prompt)
        if result is None:
            logger.warning("Perturbation failed for question: %s", question[:60])
        return result
