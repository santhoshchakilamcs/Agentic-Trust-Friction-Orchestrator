"""
Shared Anthropic Claude client for all agents.

Provides:
  - Singleton client with lazy initialization
  - Structured prompt → response helper
  - Graceful fallback flag when API key is missing
  - JSON response parsing
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client = None
_model = None


def get_client():
    """Get or create the Anthropic client singleton."""
    global _client, _model
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        _model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        if not api_key or api_key == "your-key-here":
            logger.warning("ANTHROPIC_API_KEY not set. Agents will use heuristic fallback.")
            return None
        import anthropic

        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def get_model() -> str:
    """Return the configured model name."""
    global _model
    if _model is None:
        _model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return _model


def is_available() -> bool:
    """Check if the LLM client is available (API key configured)."""
    return get_client() is not None


def call_claude(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Send a prompt to Claude and return the text response.

    Returns None if the client is not available or the call fails.
    """
    client = get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=get_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


def call_claude_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Optional[dict]:
    """
    Send a prompt to Claude and parse the response as JSON.

    The system prompt should instruct Claude to respond in JSON format.
    Returns None if the client is unavailable, call fails, or JSON is invalid.
    """
    raw = call_claude(system_prompt, user_prompt, max_tokens, temperature)
    if raw is None:
        return None
    try:
        # Handle markdown-wrapped JSON (```json ... ```)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Claude JSON response: {raw[:200]}")
        return None
