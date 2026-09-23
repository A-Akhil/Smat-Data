import os

from django.core.management.base import BaseCommand

from rag.models import ApiKeyStats

# Groq's free tier is roughly 30 requests/minute and 1000/day per key; kept as
# sane defaults here since the assessment doesn't call for tuning them further.
DEFAULTS = {"rpm_limit": 30, "rpd_limit": 1000, "tpm_limit": 100000}


class Command(BaseCommand):
    help = "Populate ApiKeyStats from GROQ_API_KEY_1.. env vars, so keys are rotated automatically."

    def handle(self, *args, **options):
        created, skipped = 0, 0
        i = 1
        while True:
            api_key = os.environ.get(f"GROQ_API_KEY_{i}")
            if not api_key:
                break
            alias = f"groq_{i}"
            _, was_created = ApiKeyStats.objects.get_or_create(
                key_alias=alias,
                defaults={"service": "groq", "api_key": api_key, **DEFAULTS},
            )
            created += was_created
            skipped += not was_created
            i += 1

        self.stdout.write(self.style.SUCCESS(f"Seeded {created} key(s), {skipped} already present."))
