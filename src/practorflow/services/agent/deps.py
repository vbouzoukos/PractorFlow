"""
Agent dependencies dataclass.

Contains shared dependencies injected into agent tool contexts.
"""

from dataclasses import dataclass
from typing import Optional, Set

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry


@dataclass
class AgentDeps:
    """Dependencies for agent tools."""

    knowledge_store: KnowledgeStore
    tool_registry: ToolRegistry
    document_scope: Optional[Set[str]] = None