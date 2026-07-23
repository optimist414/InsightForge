from __future__ import annotations

from typing import List

from ..view.models import LongTermMemory, Memory, MemoryType, ShortTermMemory


class InMemoryMemoryStore:
    """Simple memory store for MVP. Replace with DB/file-backed store later."""

    def __init__(self) -> None:
        self._memories: List[Memory] = []

    def add(self, memory: Memory) -> Memory:
        self._memories.append(memory)
        return memory

    def add_short_term(self, content: str, title: str = "", priority: int = 0) -> ShortTermMemory:
        memory = ShortTermMemory(content=content, title=title, priority=priority)
        self.add(memory)
        return memory

    def add_long_term(
        self,
        content: str,
        title: str = "",
        priority: int = 0,
        stable_key: str = "",
    ) -> LongTermMemory:
        memory = LongTermMemory(
            content=content,
            title=title,
            priority=priority,
            stable_key=stable_key,
        )
        self.add(memory)
        return memory

    def list_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[Memory]:
        memories = [memory for memory in self._memories if memory.type == memory_type]
        memories.sort(key=lambda item: (item.priority, item.created_at), reverse=True)
        return memories[:limit]

    def select_for_agent_input(self, short_limit: int = 5, long_limit: int = 5) -> List[Memory]:
        return [
            *self.list_by_type(MemoryType.SHORT_TERM, short_limit),
            *self.list_by_type(MemoryType.LONG_TERM, long_limit),
        ]
