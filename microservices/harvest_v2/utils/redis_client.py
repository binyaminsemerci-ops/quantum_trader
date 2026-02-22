"""
utils/redis_client.py
Thin wrapper around redis.Redis with retry + config from env.
"""

import os
import time
import redis


def _build_client() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db   = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
        retry_on_timeout=True,
    )


class RedisClient:
    """
    Thread-safe Redis wrapper.
    On connection failure: raises so caller can handle or exit.
    """

    def __init__(self):
        self._r = _build_client()
        self._verify()

    def _verify(self):
        retries = 5
        for attempt in range(retries):
            try:
                self._r.ping()
                return
            except redis.exceptions.ConnectionError as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"[HV2] Redis unreachable after {retries} attempts: {exc}"
                    ) from exc
                time.sleep(1.0)

    # ------------------------------------------------------------------ #
    #  Delegation — keep the surface minimal                              #
    # ------------------------------------------------------------------ #

    def keys(self, pattern: str):
        return self._r.keys(pattern)

    def hgetall(self, name: str) -> dict:
        return self._r.hgetall(name)

    def hget(self, name: str, key: str):
        return self._r.hget(name, key)

    def hset(self, name: str, mapping: dict):
        return self._r.hset(name, mapping=mapping)

    def hincrby(self, name: str, key: str, amount: int):
        return self._r.hincrby(name, key, amount)

    def hincrbyfloat(self, name: str, key: str, amount: float):
        return self._r.hincrbyfloat(name, key, amount)

    def xadd(self, stream: str, fields: dict, maxlen: int = 50_000):
        return self._r.xadd(stream, fields, maxlen=maxlen, approximate=True)

    def expire(self, name: str, seconds: int):
        return self._r.expire(name, seconds)

    @property
    def raw(self) -> redis.Redis:
        """Direct access for edge cases — use sparingly."""
        return self._r
