"""
Chat service with streaming support, RAG, and web search.

Provides a high-level service for chat workflows with:
- Session management with persistent storage
- Document upload and indexing scoped to sessions
- Knowledge search with session document scope (priority)
- Web search fallback
- Streaming response generation via agentic loop
"""

import uuid
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Set

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

from practorflow.llm import ModelPool, create_runner, StreamChunk
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.base.session import Session, Message
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.pyai import LocalLLMModel
from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
from practorflow.services.dto.chat_file import ChatFile
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("chat_service", level=appConfiguration.LoggerConfiguration.AgentLevel)


@dataclass
class ChatDeps:
    """
    Dependencies for chat agent tools.
    
    Contains knowledge store, document scope, and web search tool
    for use by registered agent tools.
    """
    
    knowledge_store: KnowledgeStore
    document_scope: Optional[Set[str]] = None
    web_search_tool: Optional[DuckDuckGoSearchTool] = None

# Chunk size for streaming
_CHUNK_SIZE = 256

# Base system instructions (always applied, not overridable by user)
_SYSTEM_INSTRUCTIONS = """<system_rules>
You are a helpful AI assistant operating within a retrieval-augmented environment with access to tools.

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

IMPORTANT OUTPUT RULES:
- Tool calls MUST NEVER be written as text.
- NEVER output JSON, tool names, arguments, or planning text.
- Tool usage must be done silently via the tool mechanism only.
- If you are deciding to call a tool, DO NOT explain or describe it.
</system_rules>
"""


def build_instructions(user_instructions: Optional[str] = None) -> str:
    """
    Build final instructions combining system rules and user instructions.
    
    Args:
        user_instructions: Optional user-provided instructions to append.
    
    Returns:
        Combined instructions string.
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


class ChatService:
    """
    High-level chat service with RAG and tool support.
    
    Manages chat sessions with:
    - Persistent session storage
    - Document upload and indexing per session
    - Knowledge search scoped to session documents
    - Web search for current information
    - Streaming response generation via agentic loop
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
            logger.info(f"[ChatService] Created new session: {session_id} for user: {user}")

        # index files
        new_file_names: List[str] = []
        if files:
            for file in files:
                doc_info = await self._index_file(file)
                session.add_document(doc_info)
                new_file_names.append(doc_info["filename"])
                logger.info(f"[ChatService] Indexed file: {doc_info['filename']} -> {doc_info['id']}")

        document_scope = self._get_document_scope(session)

        # build message
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
            )

            message_history = self._build_message_history(session)

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
            for i in range(0, len(final_text), _CHUNK_SIZE):
                yield StreamChunk(
                    text=final_text[i : i + _CHUNK_SIZE],
                    finished=False,
                )

            session.messages.append(
                Message(role="assistant", content=final_text)
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

        self._session_store.save(session)
        logger.debug(f"[ChatService] Completed response for session: {session_id}")
    
    async def delete_chat(self, session_id: str) -> bool:
        """
        Delete a chat session and its associated documents.
        
        Removes the session from storage and deletes all documents
        that were uploaded during this session from the knowledge store.
        
        Args:
            session_id: Session ID to delete.
        
        Returns:
            True if session was deleted, False if not found.
        """
        if not self._session_store.exists(session_id):
            logger.warning(f"[ChatService] Session not found for deletion: {session_id}")
            return False
        
        session = self._session_store.get(session_id)
        
        # Delete all session documents from knowledge store
        for doc in session.documents:
            doc_id = doc.get("id")
            if doc_id:
                try:
                    self._knowledge_store.delete_document(doc_id)
                    logger.debug(f"[ChatService] Deleted document: {doc_id}")
                except Exception as e:
                    logger.warning(f"[ChatService] Failed to delete document {doc_id}: {e}")
        
        # Delete session
        self._session_store.delete(session_id)
        
        logger.info(f"[ChatService] Deleted chat session: {session_id}")
        
        return True
    
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
    
    def _build_message_history(self, session: Session) -> List[ModelMessage]:
        """
        Build message history for agent context.
        
        Converts session messages to pydantic_ai ModelMessage format.
        Excludes the last user message as it will be passed directly.
        
        Args:
            session: Session containing message history.
        
        Returns:
            List of ModelMessage objects (ModelRequest/ModelResponse) for agent context.
        """
        if len(session.messages) <= 1:
            return []
        
        history: List[ModelMessage] = []
        
        for msg in session.messages[:-1]:
            # Extract text content from message
            content = msg.get_text_content()
            
            if msg.role == "user":
                # User messages become ModelRequest with UserPromptPart
                history.append(
                    ModelRequest(parts=[UserPromptPart(content=content)])
                )
            elif msg.role == "assistant":
                # Assistant messages become ModelResponse with TextPart
                history.append(
                    ModelResponse(parts=[TextPart(content=content)])
                )
        
        return history
    
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
                top_k=5,
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
            logger.debug(f"[ChatService] search_knowledge: Returning {len(results)} results")
            return formatted
        
        @agent.tool
        async def search_web(
            ctx: RunContext[ChatDeps],
            query: str,
        ) -> str:
            """
            Search the web for current information.
            
            Use this tool for:
            - Current events and news
            - Real-time information (weather, stock prices, etc.)
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
                results = ctx.deps.web_search_tool.search(query)
                
                if not results:
                    return ""
                
                # Format results
                parts = [f'Web search results for: "{query}"\n']
                
                for idx, result in enumerate(results[:5], 1):
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    url = result.get("url", "")
                    parts.append(f"{idx}. {title}\n   {snippet}\n   Source: {url}")
                
                return "\n\n".join(parts)
            
            except Exception as e:
                logger.error(f"[ChatService] Web search error: {e}")
                return ""

    async def delete_session_document(self, session_id: str, document_id: str) -> Optional[bool]:
        """
        Delete a single document from a session.
        
        Removes the document from both the session and the knowledge store.
        
        Args:
            session_id: Session ID containing the document.
            document_id: Document ID to delete.
        
        Returns:
            True if document was deleted successfully.
            False if document was not found in session.
            None if session was not found.
        """
        if not self._session_store.exists(session_id):
            logger.warning(f"[ChatService] Session not found: {session_id}")
            return None
        
        session = self._session_store.get(session_id)
        
        # Remove document from session
        removed = session.remove_document(document_id)
        
        if not removed:
            logger.warning(f"[ChatService] Document not found in session: {document_id}")
            return False
        
        # Delete from knowledge store
        try:
            self._knowledge_store.delete_document(document_id)
            logger.debug(f"[ChatService] Deleted document from knowledge store: {document_id}")
        except Exception as e:
            logger.warning(f"[ChatService] Failed to delete document from knowledge store {document_id}: {e}")
        
        # Save updated session
        self._session_store.save(session)
        
        logger.info(f"[ChatService] Deleted document {document_id} from session {session_id}")
        
        return True