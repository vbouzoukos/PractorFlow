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
from typing import AsyncIterator, List, Optional, Set

from pydantic_ai import Agent

from practorflow.llm import ModelPool, create_runner, StreamChunk
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.base.session import Session, Message
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.pyai import LocalLLMModel
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.services.dto.chat_file import ChatFile
from practorflow.logger.logger import get_logger
from practorflow.services.history.session_summary import update_session_title
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.history.builder import build_message_history
from practorflow.services.tools.registration import (
    register_default_tools,
    load_api_tools_for_user,
)
from practorflow.services.chat.chat_tools import (
    ChatDeps,
    _CHUNK_SIZE,
    _CONTEXT_EXTRACTION_PROMPT,
    build_instructions,
    _build_file_attachment_message,
    _build_session_context_message,
    _build_persona_enhanced_message,
)

logger = get_logger(
    "chat_service", level=appConfiguration.LoggerConfiguration.AgentLevel
)


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
        default_instructions: Optional[str] = None,
    ):
        """
        Initialize chat service.

        Args:
            model_pool: Model pool for acquiring LLM handles.
            model_config: Configuration for the LLM model.
            knowledge_store: Knowledge store for document storage and search.
            session_store: Session store for persisting chat sessions.
            default_instructions: Default system instructions for new sessions.
        """
        self._model_pool = model_pool
        self._model_config = model_config
        self._knowledge_store = knowledge_store
        self._session_store = session_store
        self._tool_registry = ToolRegistry()
        self._instructions = build_instructions(user_instructions=default_instructions)

        register_default_tools(self._tool_registry, self._knowledge_store)

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
        tools: List,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract persona and instructions from conversation history using the agent.

        Args:
            message_history: Conversation history in pydantic_ai format.
            model: LocalLLMModel instance.
            tools: List of Tool objects for the agent.

        Returns:
            Tuple of (persona, instructions) - either can be None if not detected.
        """
        if not message_history:
            return None, None

        agent = Agent(
            model=model,
            deps_type=ChatDeps,
            system_prompt="You are a context extraction assistant. Analyze conversation history and extract any requested persona and instructions.",
            tools=tools,
        )

        try:
            async with agent.iter(
                _CONTEXT_EXTRACTION_PROMPT,
                deps=ChatDeps(
                    knowledge_store=self._knowledge_store,
                    tool_registry=self._tool_registry,
                ),
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

        # load API tools for user and build tools list
        tools = load_api_tools_for_user(self._tool_registry, user)

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
            persona, instructions = await self._extract_context(message_history, model, tools)

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
                tools=tools,
            )

            deps = ChatDeps(
                knowledge_store=self._knowledge_store,
                tool_registry=self._tool_registry,
                document_scope=document_scope,
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