"""
Console chat application for testing ChatService.

Usage:
    python chat.py

Commands:
    /file <path1> [path2] ... <message>  - Upload files and send message
    /docs                                - List uploaded documents
    /new                                 - Start a new session
    /quit or /q or /exit                 - Exit and cleanup
"""

import asyncio
import os
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Tuple

from practorflow.llm import ModelPool
from practorflow.llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
from practorflow.services.chat import ChatService
from practorflow.settings.app_settings import appConfiguration
from practorflow.session_store.factory import create_session_store


# Console user identifier
CONSOLE_USER = "console_user"


@dataclass
class LocalChatFile:
    """Local file implementation of ChatFile protocol."""

    file: BinaryIO
    filename: str
    content_type: Optional[str] = None


def parse_input(user_input: str) -> Tuple[str, List[str]]:
    """
    Parse user input for commands and file paths.

    Args:
        user_input: Raw user input string.

    Returns:
        Tuple of (message, file_paths).
        - For /file command: (message after files, list of file paths)
        - For regular input: (full input, empty list)
    """
    user_input = user_input.strip()

    if not user_input.startswith("/file "):
        return user_input, []

    # Remove /file prefix
    remaining = user_input[6:].strip()

    if not remaining:
        return "", []

    # Parse file paths and message
    # File paths are tokens that exist as files, message is the rest
    parts = remaining.split()
    file_paths = []
    message_start_idx = 0

    for i, part in enumerate(parts):
        # Expand user home directory and normalize path
        expanded_path = os.path.expanduser(part)
        normalized_path = os.path.normpath(expanded_path)

        # Check if this part is a valid file path
        if os.path.isfile(normalized_path):
            file_paths.append(normalized_path)
            message_start_idx = i + 1
        else:
            # Once we hit a non-file token, rest is message
            break

    # Everything after file paths is the message
    message = " ".join(parts[message_start_idx:])

    # Debug: show what was parsed
    if not file_paths and remaining:
        # Check if first token looks like a path but wasn't found
        first_token = parts[0] if parts else ""
        if first_token and (
            "/" in first_token
            or "\\" in first_token
            or first_token.endswith(".pdf")
            or first_token.endswith(".txt")
        ):
            print(f"[Warning] File not found: {first_token}")

    return message, file_paths


def load_files(file_paths: List[str]) -> List[LocalChatFile]:
    """
    Load files from paths into ChatFile objects.

    Args:
        file_paths: List of file paths to load.

    Returns:
        List of LocalChatFile objects.
    """
    files = []

    for path in file_paths:
        if not os.path.isfile(path):
            print(f"[Warning] File not found: {path}")
            continue

        filename = os.path.basename(path)

        # Determine content type from extension
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".md": "text/markdown",
            ".json": "application/json",
            ".csv": "text/csv",
            ".html": "text/html",
            ".xml": "application/xml",
        }
        content_type = content_type_map.get(ext)

        # Open file in binary mode
        file_handle = open(path, "rb")

        files.append(
            LocalChatFile(
                file=file_handle,
                filename=filename,
                content_type=content_type,
            )
        )

    return files


def close_files(files: List[LocalChatFile]) -> None:
    """Close all file handles."""
    for f in files:
        try:
            f.file.close()
        except Exception:
            pass


async def handle_chat(
    service: ChatService, session_id: str, message: str, file_paths: List[str]
) -> None:
    """
    Handle a chat message with optional files.

    Args:
        service: ChatService instance.
        session_id: Current session ID.
        message: User message.
        file_paths: List of file paths to upload.
    """
    files = []

    try:
        # Load files if provided
        if file_paths:
            files = load_files(file_paths)
            if files:
                print(f"[Uploading: {', '.join(f.filename for f in files)}]")

        # Stream response
        print("Assistant: ", end="", flush=True)

        received_text = False
        async for chunk in service.chat_stream(
            session_id=session_id,
            message=message,
            user=CONSOLE_USER,
            files=files if files else None,
        ):
            if not chunk.finished:
                print(chunk.text, end="", flush=True)
                received_text = True
            else:
                if received_text:
                    print(flush=True)  # Newline after streaming completes
                if chunk.usage:
                    print(f"[Tokens: {chunk.usage.get('total_tokens', 'N/A')}]")

    finally:
        # Always close file handles
        close_files(files)


async def show_documents(service: ChatService, session_id: str) -> None:
    """Display documents in current session."""
    session = service.get_session(session_id)

    if not session or not session.documents:
        print("[No documents uploaded in this session]")
        return

    print(f"[Documents in session ({len(session.documents)}):]")
    for doc in session.list_documents():
        print(f"  - {doc['filename']} ({doc['id']})")


async def main():
    """Main console chat loop."""

    # Initialize dependencies
    print("[Initializing...]")

    config = appConfiguration.ModelConfiguration
    pool = ModelPool.get_instance(max_models=1)

    # Preload model
    print(f"[Loading model: {config.model_name}]")
    await pool.preload(config)

    knowledge_config = appConfiguration.KnowledgeChromaConfiguration
    knowledge_store = ChromaKnowledgeStore(knowledge_config)
    session_store = create_session_store()

    service = ChatService(
        model_pool=pool,
        model_config=config,
        knowledge_store=knowledge_store,
        session_store=session_store,
    )
    print("=" * 60)
    print("PractorFlow Console Chat")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /file <path1> [path2] ... <message>  - Upload files with message")
    print("  /docs                                - List uploaded documents")
    print("  /new                                 - Start new session")
    print("  /quit or /q or /exit                 - Exit and cleanup")
    print()

    # Start initial session
    current_session_id = await service.start_chat()
    print(f"[Session started: {current_session_id}]")
    print()

    # Main loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("/quit", "/q", "/exit"):
                await service.delete_chat(current_session_id)
                print("[Session deleted. Goodbye!]")
                break

            if user_input.lower() == "/new":
                # Delete old session and start new one
                await service.delete_chat(current_session_id)
                current_session_id = await service.start_chat()
                print(f"[New session started: {current_session_id}]")
                continue

            if user_input.lower() == "/docs":
                await show_documents(service, current_session_id)
                continue

            # Parse input for files and message
            message, file_paths = parse_input(user_input)

            if not message:
                print("[Error: Please provide a message]")
                continue

            # Handle chat
            await handle_chat(service, current_session_id, message, file_paths)
            print()

        except KeyboardInterrupt:
            print("\n[Interrupted]")
            await service.delete_chat(current_session_id)
            print("[Session deleted. Goodbye!]")
            break

        except Exception as e:
            print(f"\n[Error: {e}]")
            continue

    # Cleanup
    await pool.unload_all()


if __name__ == "__main__":
    asyncio.run(main())