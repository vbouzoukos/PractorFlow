"""
Chat service with streaming support, RAG, and web search.

Provides a high-level service for chat workflows with:
- Session management with persistent storage
- Document upload and indexing scoped to sessions
- Knowledge search with session document scope (priority)
- Web search fallback
- Streaming response generation via agentic loop
- Persona extraction and persistence across conversation
"""

import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Set

from pydantic_ai import Agent, RunContext

from practorflow.llm import ModelPool, create_runner, StreamChunk
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.base.session import Session, Message
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.pyai import LocalLLMModel
from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
from practorflow.services.dto.chat_file import ChatFile
from practorflow.logger.logger import get_logger
from practorflow.services.history.session_summary import update_session_title
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.history.builder import build_message_history

logger = get_logger(
    "chat_service", level=appConfiguration.LoggerConfiguration.AgentLevel
)


@dataclass
class ChatDeps:
    """
    Dependencies for chat agent tools.

    Contains knowledge store, document scope, web search tool,
    and extracted persona for use by registered agent tools.
    """

    knowledge_store: KnowledgeStore
    document_scope: Optional[Set[str]] = None
    web_search_tool: Optional[DuckDuckGoSearchTool] = None
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


class ChatService:
    """
    High-level chat service with RAG and tool support.

    Manages chat sessions with:
    - Persistent session storage
    - Document upload and indexing per session
    - Knowledge search scoped to session documents
    - Web search for current information
    - Streaming response generation via agentic loop
    - Persona extraction and persistence
    """

    def __init__(
        self,
        model_pool: ModelPool,
        model_config: LLMConfig,
        knowledge_store: KnowledgeStore,
        session_store: SessionStore,
        web_search_tool: Optional[DuckDuckGoSearchTool] = None,
        default_instructions: Optional[str] = None,
    ):
        """
        Initialize chat service.

        Args:
            model_pool: Model pool for acquiring LLM handles.
            model_config: Configuration for the LLM model.
            knowledge_store: Knowledge store for document storage and search.
            session_store: Session store for persisting chat sessions.
            web_search_tool: Optional web search tool (defaults to DuckDuckGo).
            default_instructions: Default system instructions for new sessions.
        """
        self._model_pool = model_pool
        self._model_config = model_config
        self._knowledge_store = knowledge_store
        self._session_store = session_store
        self._web_search_tool = web_search_tool or DuckDuckGoSearchTool()
        self._instructions = build_instructions(user_instructions=default_instructions)

        logger.info("[ChatService] Initialized")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{uuid.uuid4().hex}"

    async def start_chat(self) -> str:
        """
        Start a new chat session.

        Generates a unique session ID and returns it. The session is not
        persisted until the first message is sent via chat_stream.

        Returns:
            The generated session_id.
        """
        session_id = self._generate_session_id()

        logger.info(f"[ChatService] Generated session ID: {session_id}")

        return session_id

    async def _extract_context(
        self,
        message_history: List,
        model: LocalLLMModel,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract persona and instructions from conversation history using the agent.

        Args:
            message_history: Conversation history in pydantic_ai format.
            model: LocalLLMModel instance.

        Returns:
            Tuple of (persona, instructions) - either can be None if not detected.
        """
        if not message_history:
            return None, None

        agent = Agent(
            model=model,
            deps_type=ChatDeps,
            system_prompt="You are a context extraction assistant. Analyze conversation history and extract any requested persona and instructions.",
        )

        try:
            async with agent.iter(
                _CONTEXT_EXTRACTION_PROMPT,
                deps=ChatDeps(knowledge_store=self._knowledge_store),
                message_history=message_history,
            ) as agent_run:
                async for node in agent_run:
                    logger.debug(
                        "[ChatService][ContextExtract] node=%s",
                        getattr(node, "name", type(node).__name__),
                    )

                if agent_run.result:
                    result = agent_run.result.output or ""
                    result = result.strip()

                    persona = None
                    instructions = None

                    # Parse persona
                    for line in result.split("\n"):
                        line = line.strip()
                        if line.upper().startswith("PERSONA:"):
                            value = line[8:].strip()
                            if value.upper() != "NONE" and value:
                                persona = value
                            break

                    # Parse instructions between markers
                    if "INSTRUCTIONS_START" in result and "INSTRUCTIONS_END" in result:
                        start_idx = result.find("INSTRUCTIONS_START") + len("INSTRUCTIONS_START")
                        end_idx = result.find("INSTRUCTIONS_END")
                        if start_idx < end_idx:
                            instructions_text = result[start_idx:end_idx].strip()
                            if instructions_text.upper() != "NONE" and instructions_text:
                                instructions = instructions_text

                    if persona:
                        logger.info(f"[ChatService] Extracted persona: {persona}")
                    if instructions:
                        logger.info(f"[ChatService] Extracted instructions: {instructions[:100]}...")

                    return persona, instructions

        except Exception as e:
            logger.warning(f"[ChatService] Context extraction failed: {e}")
            return None, None

        return None, None

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        user: str,
        files: Optional[List[ChatFile]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a message and stream the response.

        Processes the user message, optionally indexes uploaded files,
        and streams the assistant's response using the configured tools.
        Creates the session on first call if it doesn't exist.

        Uses two-step agentic approach:
        1. Extract persona from conversation history
        2. Generate response with persona context

        Args:
            session_id: Session ID for the chat.
            message: User message text.
            user: User identifier for the session.
            files: Optional list of files to upload and index for this session.

        Yields:
            StreamChunk objects with response text and metadata.
        """
        # get or create session
        if self._session_store.exists(session_id):
            session = self._session_store.get(session_id)
        else:
            session = Session(
                session_id=session_id,
                instructions=self._instructions,
                user=user,
            )
            logger.info(
                f"[ChatService] Created new session: {session_id} for user: {user}"
            )

        # index files
        new_file_names: List[str] = []
        if files:
            for file in files:
                doc_info = await self._index_file(file)
                session.add_document(doc_info)
                new_file_names.append(doc_info["filename"])
                logger.info(
                    f"[ChatService] Indexed file: {doc_info['filename']} -> {doc_info['id']}"
                )

        document_scope = self._get_document_scope(session)

        # build base message
        if new_file_names:
            enhanced_message = _build_file_attachment_message(new_file_names, message)
        elif session.documents:
            enhanced_message = _build_session_context_message(
                [doc.get("filename", "unknown") for doc in session.documents],
                message,
            )
        else:
            enhanced_message = message

        # append user message
        session.messages.append(Message(role="user", content=message))

        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""

        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=session.instructions)

            message_history = build_message_history(session)

            # Step 1: Extract persona and instructions from history
            persona, instructions = await self._extract_context(message_history, model)

            # Step 2: Enhance message with persona/instructions if detected
            if persona or instructions:
                enhanced_message = _build_persona_enhanced_message(
                    persona, instructions, enhanced_message
                )
                logger.debug(f"[ChatService] Applied context - persona: {persona}, instructions: {instructions}")

            # Step 3: Generate response with full context
            agent = Agent(
                model=model,
                deps_type=ChatDeps,
                system_prompt=session.instructions,
            )
            self._register_tools(agent)

            deps = ChatDeps(
                knowledge_store=self._knowledge_store,
                document_scope=document_scope,
                web_search_tool=self._web_search_tool,
                persona=persona,
            )

            async with agent.iter(
                enhanced_message,
                deps=deps,
                message_history=message_history,
            ) as agent_run:
                async for node in agent_run:
                    logger.debug(
                        "[ChatService][AgentIter] node=%s",
                        getattr(node, "name", type(node).__name__),
                    )

                if agent_run.result:
                    usage = agent_run.usage()
                    total_input_tokens = usage.input_tokens
                    total_output_tokens = usage.output_tokens
                    final_text = agent_run.result.output or ""

        # stream final result
        if final_text:
            # generate session title if not already set
            session.messages.append(Message(role="assistant", content=final_text))
            if session.title is None:
                await update_session_title(
                    session=session,
                    model_pool=self._model_pool,
                    model_config=self._model_config,
                )

            self._session_store.save(session)

            for i in range(0, len(final_text), _CHUNK_SIZE):
                yield StreamChunk(
                    text=final_text[i : i + _CHUNK_SIZE],
                    finished=False,
                )

        # final chunk
        yield StreamChunk(
            text="",
            finished=True,
            finish_reason="stop",
            usage={
                "prompt_tokens": total_input_tokens,
                "completion_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
        )

        logger.debug(f"[ChatService] Completed response for session: {session_id}")

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session object or None if not found.
        """
        if not self._session_store.exists(session_id):
            return None
        return self._session_store.get(session_id)

    async def _index_file(self, file: ChatFile) -> dict:
        """
        Index a file into the knowledge store.

        Args:
            file: File to index.

        Returns:
            Document info dict with id, filename, etc.
        """
        doc_info = self._knowledge_store.add_document_from_stream(
            file_stream=file.file,
            filename=file.filename,
            mime_type=file.content_type,
        )
        return doc_info

    def _get_document_scope(self, session: Session) -> Optional[Set[str]]:
        """
        Get document IDs from session for scoped search.

        Args:
            session: Session to extract document IDs from.

        Returns:
            Set of document IDs or None if no documents.
        """
        if not session.documents:
            return None

        return {doc["id"] for doc in session.documents if "id" in doc}

    def _register_tools(self, agent: Agent) -> None:
        """
        Register tools with the agent.

        Args:
            agent: Agent to register tools with.
        """

        @agent.tool
        async def search_knowledge(
            ctx: RunContext[ChatDeps],
            query: str,
        ) -> str:
            """
            Search the knowledge base for relevant information from uploaded documents.

            IMPORTANT: You MUST call this tool when the user has attached files or
            asks questions about documents. This is the ONLY way to access file contents.

            Args:
                query: Search query text to find relevant content in documents.

            Returns:
                Relevant text from documents or message if none found.
            """
            logger.debug(f"[ChatService] search_knowledge called with query: {query}")

            results = ctx.deps.knowledge_store.search_scoped(
                query=query,
                top_k=10,
                document_ids=ctx.deps.document_scope,
            )

            if not results:
                logger.debug("[ChatService] search_knowledge: No results found")
                return ""

            # Format results for LLM context
            parts = []
            parts.append(f'Search results for: "{query}"')
            parts.append(f"Found {len(results)} relevant section(s):\n")

            for idx, result in enumerate(results, 1):
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                filename = result.get("filename") or metadata.get("filename", "unknown")
                similarity = result.get("similarity", 0.0)

                header = f"--- Section {idx} (Source: {filename}, Relevance: {similarity:.2f}) ---"
                parts.append(f"{header}\n{text}")

            formatted = "\n\n".join(parts)
            logger.debug(
                f"[ChatService] search_knowledge: Returning {len(results)} results"
            )
            return formatted

        @agent.tool
        async def search_web(ctx: RunContext[ChatDeps], query: str) -> str:
            """
            Search the web for current information.

            Use this tool for:
            - Current events and news
            - Real-time information
            - Information that may have changed since training

            Args:
                query: Search query for web search.

            Returns:
                Web search results or error message.
            """
            logger.debug(f"[ChatService] search_web called with query: {query}")

            if not ctx.deps.web_search_tool:
                return "Web search is not available."

            try:
                result = ctx.deps.web_search_tool.execute(query=query)

                if not result.success:
                    logger.warning(f"[ChatService] Web search failed: {result.error}")
                    return ""

                if not result.data:
                    return ""

                # result.data is already formatted by _format_results in the tool
                return result.data

            except Exception as e:
                logger.error(f"[ChatService] Web search error: {e}")
                return ""