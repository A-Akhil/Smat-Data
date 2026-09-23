"""API key rotation and rate-limit tracking for the Groq LLM client.

Same approach as the calquity project: multiple keys per service, rate limits
tracked in the database, atomic selection so concurrent requests don't
oversubscribe a key.
"""

import logging

from django.db import transaction
from django.utils import timezone

from rag.models import ApiKeyStats

logger = logging.getLogger(__name__)


class RateLimitExhausted(Exception):
    def __init__(self, service: str):
        self.service = service
        super().__init__(f"All API keys for service '{service}' have exhausted their rate limits.")


def get_available_key(service: str) -> tuple[str, str]:
    now = timezone.now()

    with transaction.atomic():
        keys = ApiKeyStats.objects.select_for_update().filter(service=service, is_active=True)

        for key in keys:
            if key.last_reset_minute is None or (now - key.last_reset_minute).total_seconds() > 60:
                key.requests_this_minute = 0
                key.tokens_this_minute = 0
                key.last_reset_minute = now

            if key.last_reset_day is None or key.last_reset_day.date() < now.date():
                key.requests_today = 0
                key.last_reset_day = now

            if (
                key.requests_this_minute < key.rpm_limit
                and key.requests_today < key.rpd_limit
                and key.tokens_this_minute < key.tpm_limit
            ):
                key.requests_this_minute += 1
                key.last_request_at = now
                key.save()
                return (key.key_alias, key.api_key)

            key.save()

    logger.warning("All API keys exhausted for service '%s'.", service)
    raise RateLimitExhausted(service)


def record_token_usage(key_alias: str, tokens_used: int) -> None:
    with transaction.atomic():
        try:
            key = ApiKeyStats.objects.select_for_update().get(key_alias=key_alias)
            key.tokens_this_minute += tokens_used
            key.save()
        except ApiKeyStats.DoesNotExist:
            logger.error("Cannot record token usage: key alias '%s' not found.", key_alias)


def penalize_key(key_alias: str) -> None:
    with transaction.atomic():
        try:
            key = ApiKeyStats.objects.select_for_update().get(key_alias=key_alias)
            key.requests_this_minute = key.rpm_limit
            key.tokens_this_minute = key.tpm_limit
            key.save(update_fields=["requests_this_minute", "tokens_this_minute"])
        except ApiKeyStats.DoesNotExist:
            pass
