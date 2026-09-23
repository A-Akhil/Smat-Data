from django.db import models


class ApiKeyStats(models.Model):
    """Tracks per-key rate-limit usage so multiple Groq keys can be rotated
    between, the same approach used in the calquity project for its Groq/Gemini
    keys."""

    service = models.CharField(max_length=20)
    key_alias = models.CharField(max_length=50, unique=True)
    api_key = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True)

    rpm_limit = models.IntegerField(default=30)
    rpd_limit = models.IntegerField(default=1000)
    tpm_limit = models.IntegerField(default=100000)

    requests_this_minute = models.IntegerField(default=0)
    requests_today = models.IntegerField(default=0)
    tokens_this_minute = models.IntegerField(default=0)

    last_reset_minute = models.DateTimeField(null=True, blank=True)
    last_reset_day = models.DateTimeField(null=True, blank=True)
    last_request_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.service}:{self.key_alias}"
