"""Groq chat-completion client with automatic key rotation."""

import logging
import os
import time

from groq import Groq

from rag.api_manager import get_available_key, penalize_key, record_token_usage

logger = logging.getLogger(__name__)

# "groq/compound" (used in the calquity project) is no longer available on
# this account's plan; openai/gpt-oss-120b is Groq-hosted, tool-use capable,
# and currently accessible. Configurable via env in case that changes again.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


class GroqClient:
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion request, rotating keys on rate-limit errors.

        Returns the assistant's reply text.
        """
        last_exception = None

        for attempt in range(max_retries):
            key_alias, api_key = get_available_key("groq")
            client = Groq(api_key=api_key)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                total_tokens = 0
                if getattr(response, "usage", None) is not None:
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0
                record_token_usage(key_alias, total_tokens)
                return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    logger.warning("Groq key '%s' hit a rate limit, rotating.", key_alias)
                    penalize_key(key_alias)
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(f"Groq API failed after {max_retries} attempts. Last error: {last_exception}")
