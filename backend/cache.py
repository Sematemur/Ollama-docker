"""Redis cache helpers for LLM outputs."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import redis

from config import settings

LOGGER = logging.getLogger("chat.cache")


def normalize_question(question: str) -> str:
    """Normalize user input for stable cache keys."""
    return " ".join(question.strip().lower().split())


class ResponseCache:
    """Thin Redis wrapper for storing LLM responses."""

    def __init__(self) -> None:
        self.enabled = settings.enable_response_cache
        self.redis_url = settings.redis_url
        self.ttl_seconds = settings.redis_cache_ttl_seconds
        self.max_connections = settings.redis_max_connections
        self.sensitive_patterns = [
            pattern.strip().lower()
            for pattern in settings.cache_sensitive_patterns.split(",")
            if pattern.strip()
        ]
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None

    def _get_client(self) -> Optional[redis.Redis]:
        if not self.enabled:
            return None

        if self._client is not None:
            return self._client

        try:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=self.max_connections,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._client.ping()
        except redis.RedisError as exc:
            LOGGER.warning("Redis unavailable, caching disabled for this request: %s", exc)
            self._client = None
            if self._pool:
                self._pool.disconnect()
                self._pool = None

        return self._client

    @staticmethod
    def build_key(question: str) -> str:
        normalized = normalize_question(question)
        question_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"llm_response:v1:{question_hash}"

    def _contains_sensitive_content(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in self.sensitive_patterns)

    def get(self, question: str) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None

        try:
            key = self.build_key(question)
            return client.get(key)
        except redis.RedisError as exc:
            LOGGER.warning("Cache read failed: %s", exc)
            return None

    def set(self, question: str, response: str) -> None:
        client = self._get_client()
        if client is None:
            return

        if self._contains_sensitive_content(response):
            LOGGER.debug("Sensitive content detected, skipping cache write.")
            return

        try:
            key = self.build_key(question)
            client.setex(key, self.ttl_seconds, response)
        except redis.RedisError as exc:
            LOGGER.warning("Cache write failed: %s", exc)

    def close(self) -> None:
        """Close active Redis client and release pooled connections."""
        if self._client is not None:
            try:
                self._client.close()
            except redis.RedisError:
                pass
            self._client = None

        if self._pool is not None:
            self._pool.disconnect()
            self._pool = None
