"""
Base Source Interface
====================
Abstract interface all content sources implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from backend.models import ContentItem


class BaseSource(ABC):
    """Abstract base for all content sources."""

    source_type: str = ""

    @abstractmethod
    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch content items from this source."""
        ...
