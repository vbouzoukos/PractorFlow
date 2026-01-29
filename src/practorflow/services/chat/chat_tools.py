"""
Chat service tools and dependencies.

Provides ChatDeps, prompts, and helper functions for chat service.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from practorflow.services.tools.deps import ToolsDeps


@dataclass
class ChatDeps(ToolsDeps):
    """
    Dependencies for chat agent tools.

    Extends ToolsDeps with chat-specific fields.
    """

    persona: Optional[str] = field(default=None)


# Chunk size for streaming
_CHUNK_SIZE = 256

# Prompt for persona and instructions extraction from conversation history
_CONTEXT_EXTRACTION_PROMPT = """Analyze the conversation history and extract TWO things:

1. PERSONA: Any requested persona, character, tone, or role for the assistant.
   Look for patterns like:
   - "act as...", "be a...", "pretend you are...", "respond like..."
   - "you are a...", "play the role of..."
   - Character requests: pirate, teacher, expert, conspiracist, etc.

2. INSTRUCTIONS: Any custom instructions the user has explicitly given.
   Extract the EXACT user instructions as they were written - do not summarize or paraphrase.
   Look for messages where the user gives explicit directives about how to respond.

Respond in EXACTLY this format:
PERSONA: <persona description or NONE>
INSTRUCTIONS_START
<exact user instructions copied verbatim, or NONE if no instructions found>
INSTRUCTIONS_END

Examples:

Example 1 - User said "act as a pirate":
PERSONA: pirate
INSTRUCTIONS_START
NONE
INSTRUCTIONS_END

Example 2 - User said "You are a formal assistant. Always respond in bullet points and include references.":
PERSONA: formal assistant
INSTRUCTIONS_START
Always respond in bullet points and include references.
INSTRUCTIONS_END

Example 3 - User said "Remember these rules: 1. Be concise 2. Use examples 3. Avoid jargon":
PERSONA: NONE
INSTRUCTIONS_START
1. Be concise 2. Use examples 3. Avoid jargon
INSTRUCTIONS_END

Example 4 - No persona or instructions found:
PERSONA: NONE
INSTRUCTIONS_START
NONE
INSTRUCTIONS_END

Extract and respond now."""

# Base system instructions (always applied, not overridable by user)
_SYSTEM_INSTRUCTIONS = """<system_rules>
You are a helpful AI assistant operating within a retrieval-augmented environment with access to tools.

PERSONA AND STYLE ADAPTATION:
- BEFORE responding, scan the conversation history for any user requests to adopt a specific persona, tone, style, or character (e.g., "act as...", "be a...", "pretend you are...", "respond like...").
- If a persona was requested earlier in the conversation, you MUST maintain it for ALL subsequent responses until the user explicitly changes or cancels it.
- User-defined persona preferences (e.g., "be formal", "act as a pirate", "respond like a teacher", "act as a conspiracist") take priority over default assistant behavior.
- When adopting a persona, DO NOT break character with disclaimers, caveats, objective commentary, or "balanced perspectives" unless the user explicitly asks for them.
- Stay fully in character. Do not add phrases like "It's important to note...", "From a scientific standpoint...", or "However, in reality..." when roleplaying a persona.
- The persona applies to your ENTIRE response, not just the opening. Do not revert to default behavior mid-response.
- If no persona is specified anywhere in the conversation history, default to being helpful, clear, and professional.

CRITICAL TOOL USAGE RULES - YOU MUST FOLLOW THESE:
1. You have access to two tools: search_knowledge and search_web.
2. When the user attaches files or documents, you MUST call search_knowledge with a relevant query BEFORE responding. This is MANDATORY - do not skip this step.
3. You cannot see file contents directly. The ONLY way to access file content is by calling search_knowledge.
4. If you respond about files without first calling search_knowledge, your response will be incorrect.

MANDATORY TOOL CALLING PROTOCOL:
- Files/documents attached or referenced → MUST call search_knowledge first. No exceptions.
- Questions about attached documents → MUST call search_knowledge first. Do not guess or assume content.
- Current events, news, live data → Call search_web.
- General knowledge (no files, no web request) → Respond from training knowledge.

RESPONSE RULES:
1. NEVER mention tool names (search_knowledge, search_web) to the user.
2. NEVER say "I will search" or "Let me search". Just call the tool silently.
3. Present information naturally as if you already know it.

CONVERSATION AWARENESS:
4. Review conversation history - do NOT repeat information already provided.
5. For follow-up questions ("what else", "anything more", "besides that"), focus ONLY on NEW information.
6. If search results contain the same information as before, tell the user no additional information is available.
7. Use different search queries for follow-ups to find new content.

IMPORTANT OUTPUT RULES:
- Tool calls MUST NEVER be written as text.
- NEVER output JSON, tool names, arguments, or planning text.
- Tool usage must be done silently via the tool mechanism only.
- If you are deciding to call a tool, DO NOT explain or describe it.
</system_rules>
"""


def build_instructions(user_instructions: Optional[str] = None) -> str:
    """
    Build complete system instructions.

    Combines base system rules with optional user instructions.

    Args:
        user_instructions: Optional additional instructions from user.

    Returns:
        Complete system instructions string.
    """
    if user_instructions:
        return f"{_SYSTEM_INSTRUCTIONS}\n\n<user_instructions>\n{user_instructions}\n</user_instructions>"
    return _SYSTEM_INSTRUCTIONS


def _build_file_attachment_message(file_names: List[str], user_message: str) -> str:
    """
    Build enhanced message when files are newly attached.

    Args:
        file_names: List of attached file names.
        user_message: Original user message.

    Returns:
        Enhanced message with file context.
    """
    files_str = ", ".join(file_names)
    return f"""[User attached files: {files_str}]

IMPORTANT: You MUST call search_knowledge with a relevant query to access the file contents before responding.

User message: {user_message}"""


def _build_session_context_message(file_names: List[str], user_message: str) -> str:
    """
    Build enhanced message when session has existing documents.

    Args:
        file_names: List of document names in session.
        user_message: Original user message.

    Returns:
        Enhanced message with session context.
    """
    files_str = ", ".join(file_names)
    return f"""[Session has documents: {files_str}]

If the user's question relates to these documents, call search_knowledge first.

User message: {user_message}"""


def _build_persona_enhanced_message(persona: str, instructions: str, user_message: str) -> str:
    """
    Build message with persona and instructions reminder injected.

    Args:
        persona: Extracted persona description (or None).
        instructions: Extracted instructions (or None).
        user_message: Original or enhanced user message.

    Returns:
        Message with persona/instructions reminder prepended.
    """
    parts = []
    
    if persona:
        parts.append(f'PERSONA ACTIVE: You are acting as "{persona}". Stay fully in character for your entire response. Do not break character with disclaimers or objective commentary.')
    
    if instructions:
        parts.append(f'USER INSTRUCTIONS: {instructions}')
    
    if parts:
        context = "[" + " | ".join(parts) + "]"
        return f"{context}\n\n{user_message}"
    
    return user_message