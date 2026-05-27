"""
rop/llm/client.py

Unified LLM API client with retry logic and rate limiting.
API keys are read exclusively from environment variables — never hardcoded.
"""

import os
import time
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the OpenAI client that adds:
    - Automatic retries with exponential back-off
    - Per-request rate-limit sleep
    - A single place to swap models or base URLs
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_retries: int = 3,
        retry_interval: float = 5.0,
        request_interval: float = 1.0,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Run: export OPENAI_API_KEY='sk-...'"
            )

        base_url = os.environ.get("OPENAI_BASE_URL")  # optional proxy
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.request_interval = request_interval

    def complete(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Send a single-turn chat completion request.

        Args:
            prompt: The user message.
            model:  Override the default model for this call.

        Returns:
            The assistant's reply, or None if all retries failed.
        """
        target_model = model or self.model

        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_interval)
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for model '%s': %s",
                    attempt,
                    self.max_retries,
                    target_model,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_interval * attempt)

        logger.error("All %d attempts failed. Returning None.", self.max_retries)
        return None
