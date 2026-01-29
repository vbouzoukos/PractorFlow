"""
Agent dependencies dataclass.

Contains shared dependencies injected into agent tool contexts.
"""

from dataclasses import dataclass

from practorflow.services.tools.deps import ToolsDeps


@dataclass
class AgentDeps(ToolsDeps):
    """Dependencies for agent tools."""
