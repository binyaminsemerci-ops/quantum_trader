"""Base class for Redis stream message contracts."""

from __future__ import annotations
from typing import Dict
from pydantic import BaseModel, ConfigDict


class StreamEvent(BaseModel):
    """Base class for typed Redis stream events.

    All field values are stored as strings in Redis.
    Subclasses define the schema; to_redis()/from_redis()
    handle serialization.
    """

    model_config = ConfigDict(extra="allow")

    def to_redis(self) -> Dict[str, str]:
        """Convert to Dict[str, str] suitable for redis.xadd()."""
        return {k: str(v) for k, v in self.model_dump().items() if v is not None}

    @classmethod
    def from_redis(cls, fields: Dict) -> "StreamEvent":
        """Parse a Redis XREADGROUP result (bytes keys/values).

        Decodes bytes → str automatically.
        """
        decoded = {}
        for k, v in fields.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            decoded[key] = val
        return cls.model_validate(decoded)
