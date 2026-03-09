"""
PII Masker: Middleware that redacts sensitive data before it reaches any LLM or logging layer.

Uses SHA-256 hashing instead of simple redaction so that:
  - Same input always produces the same hash (correlatable across records)
  - One-way: cannot reverse the hash back to the original PII
  - Prefixed with field type for readability (e.g., "ip_a3f2b8...")
"""

import hashlib
import os
import re
from typing import Dict, Any


# Salt loaded once at import time; set PII_HASH_SALT in .env for production
_SALT = os.getenv("PII_HASH_SALT", "agentic-orchestrator-default-salt").encode()


class PIIMasker:
    """Scrubs personally identifiable information from data before external processing."""

    # Patterns to detect and mask in free-text strings
    PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "phone": re.compile(r"\+?1?\d{9,15}"),
        "ssn": re.compile(r"\d{3}-\d{2}-\d{4}"),
        "card_number": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    }

    # Fields whose entire value should be hashed (not pattern-searched)
    SENSITIVE_FIELDS = {"ip_address", "device_id", "recipient_name", "name"}

    # ── Hashing helpers ─────────────────────────────────────────────────────

    @classmethod
    def _hash(cls, value: str, prefix: str = "") -> str:
        """Produce a salted SHA-256 hash of *value*, truncated to 16 hex chars.

        Returns e.g. ``"ip_a3f2b8c1d9e04567"`` when prefix="ip".
        """
        digest = hashlib.sha256(_SALT + value.encode()).hexdigest()[:16]
        return f"{prefix}_{digest}" if prefix else digest

    # ── Public API ──────────────────────────────────────────────────────────

    @classmethod
    def mask_string(cls, text: str) -> str:
        """Hash PII patterns found inside a free-text string."""
        masked = text
        for name, pattern in cls.PATTERNS.items():
            masked = pattern.sub(
                lambda m, n=name: cls._hash(m.group(), prefix=n),
                masked,
            )
        return masked

    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the dict with sensitive fields hashed."""
        masked = {}
        for key, value in data.items():
            if key in cls.SENSITIVE_FIELDS:
                if isinstance(value, str):
                    masked[key] = cls._hash(value, prefix=key)
                else:
                    masked[key] = cls._hash(str(value), prefix=key)
            elif isinstance(value, str):
                masked[key] = cls.mask_string(value)
            elif isinstance(value, dict):
                masked[key] = cls.mask_dict(value)
            else:
                masked[key] = value
        return masked

    @classmethod
    def mask_for_logging(cls, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state dict for safe logging / external transmission."""
        return cls.mask_dict(state_dict)

