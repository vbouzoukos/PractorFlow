"""
Base tool dependencies dataclass.

Provides common dependencies for tool execution across services.
"""

from dataclasses import dataclass
from typing import Optional, Set

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry


@dataclass
class ToolsDeps:
    """
    Base dependencies for tool execution.
    
    Inherited by AgentDeps and ChatDeps to provide common
    tool infrastructure.
    """

    knowledge_store: KnowledgeStore
    tool_registry: ToolRegistry
    document_scope: Optional[Set[str]] = None