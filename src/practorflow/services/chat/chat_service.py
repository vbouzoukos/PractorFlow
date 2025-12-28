"""
Chat service with streaming support, RAG, and web search.

Provides a high-level service for chat workflows with:
- Session management with persistent storage
- Document upload and indexing scoped to sessions
- Knowledge search with session document scope (priority)
- Web search fallback
- Streaming response generation
"""

import uuid
from dataclasses import dataclass
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


class ChatService:
    """
    Chat service with streaming, RAG, and web search support.
    
    Manages chat sessions with document context and provides
    streaming responses using local LLM models.
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
        self._instructions = default_instructions or (
            "You are a helpful AI assistant. "
            "When the user attaches files, you MUST use the search_knowledge tool to read and retrieve their content before answering. "
            "Use the search_knowledge tool to find information from uploaded documents. "
            "Use the search_web tool only when the user explicitly asks for current events, news, or web information."
        )
        
        self.session: Optional[Session] = None
        
        logger.info("[ChatService] Initialized")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{uuid.uuid4().hex}"
    
    async def start_chat(
        self,
        instructions: Optional[str] = None,
    ) -> str:
        """
        Start a new chat session.
        
        Generates a unique session ID and returns a Session object.
        The session is persisted when chat_stream is first called.
        
        Args:
            instructions: Optional system instructions for the assistant.
                         Uses default instructions if not provided.
        
        Returns:
            The created Session object with generated session_id.
        """
        session_id = self._generate_session_id()
        
        self.session = Session(
            session_id=session_id,
            instructions=instructions or self._instructions,
        )
        
        logger.info(f"[ChatService] Started chat session: {session_id}")
        
        return session_id
    
    async def chat_stream(
        self,
        session_id: str,
        message: str,
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
            files: Optional list of files to upload and index for this session.
        
        Yields:
            StreamChunk objects with response text and metadata.
        """
        if not self.session:
            raise ValueError("No active session. Call start_chat first.")
        
        session = self.session
        
        # Track newly uploaded file names
        new_file_names: List[str] = []
        
        # Index files if provided
        if files:
            for file in files:
                doc_info = await self._index_file(file)
                session.add_document(doc_info)
                new_file_names.append(doc_info['filename'])
                logger.info(f"[ChatService] Indexed file: {doc_info['filename']} -> {doc_info['id']}")
            self._session_store.save(session)
        
        # Get document scope from session
        document_scope = self._get_document_scope(session)
        
        # Build message with file attachment notification for the agent
        if new_file_names:
            enhanced_message = f"[User attached files: {', '.join(new_file_names)}]\n\n{message}"
        else:
            enhanced_message = message
        
        # Add user message to session (store original, not enhanced)
        user_message = Message(role="user", content=message)
        session.messages.append(user_message)
        
        # Accumulate response for session storage
        accumulated_response = []
        usage = None
        
        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=session.instructions)
            
            # Create agent with tools
            agent = Agent(
                model=model,
                deps_type=ChatDeps,
                system_prompt=session.instructions,
            )
            
            # Register tools
            self._register_tools(agent)
            
            # Create dependencies
            deps = ChatDeps(
                knowledge_store=self._knowledge_store,
                document_scope=document_scope,
                web_search_tool=self._web_search_tool,
            )
            
            # Build message history for context
            message_history = self._build_message_history(session)
            
            logger.debug(f"[ChatService] Streaming response for session: {session_id}")
            
            # Stream response
            async with agent.run_stream(
                enhanced_message,
                deps=deps,
                message_history=message_history,
            ) as response:
                async for text in response.stream_text():
                    accumulated_response.append(text)
                    yield StreamChunk(text=text, finished=False)
                
                # Extract usage if available
                try:
                    usage_obj = response.usage()
                    usage = {
                        "prompt_tokens": getattr(usage_obj, 'input_tokens', 0),
                        "completion_tokens": getattr(usage_obj, 'output_tokens', 0),
                        "total_tokens": getattr(usage_obj, 'input_tokens', 0) + getattr(usage_obj, 'output_tokens', 0),
                    }
                except Exception:
                    pass
            
            # Yield final chunk BEFORE exiting model context
            yield StreamChunk(
                text="",
                finished=True,
                finish_reason="stop",
                usage=usage,
            )
        
        # Store assistant response in session (after model released)
        full_response = "".join(accumulated_response)
        assistant_message = Message(role="assistant", content=full_response)
        session.messages.append(assistant_message)
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
    
    def _build_message_history(self, session: Session) -> list:
        """
        Build message history for agent context.
        
        Excludes the last user message as it will be passed directly.
        
        Args:
            session: Session containing message history.
        
        Returns:
            List of message dicts for agent context.
        """
        if len(session.messages) <= 1:
            return []
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages[:-1]
        ]
    
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
            Search the knowledge base for relevant information.
            
            Use this tool to find information from uploaded documents.
            
            Args:
                query: Search query text.
            
            Returns:
                Relevant text from documents or message if none found.
            """
            results = ctx.deps.knowledge_store.search(
                query=query,
                top_k=5,
                document_ids=ctx.deps.document_scope,
            )
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            formatted = []
            for r in results:
                source = r.get("metadata", {}).get("filename", "Unknown")
                text = r.get("text", "")
                formatted.append(f"[Source: {source}]\n{text}")
            
            return "\n\n---\n\n".join(formatted)
        
        @agent.tool
        async def search_web(
            ctx: RunContext[ChatDeps],
            query: str,
        ) -> str:
            """
            Search the web for current information.
            
            Use this tool only when explicitly asked for current events,
            news, or information not in uploaded documents.
            
            Args:
                query: Search query text.
            
            Returns:
                Web search results or error message.
            """
            if not ctx.deps.web_search_tool:
                return "Web search is not available."
            
            try:
                results = ctx.deps.web_search_tool.search(query)
                return results
            except Exception as e:
                logger.warning(f"[ChatService] Web search failed: {e}")
                return f"Web search failed: {str(e)}"