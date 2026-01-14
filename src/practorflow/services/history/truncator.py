from logging import Logger
from typing import Optional

from practorflow.llm.base.session_store import SessionStore


async def truncate_messages(
    session_id: str, from_index: int, session_store: SessionStore
) -> Optional[int]:
    """
    Truncate messages from a given index onwards.

    Used for edit-and-regenerate functionality where the user
    edits a message and all subsequent messages are removed.

    Args:
        session_id: Session ID to truncate messages from.
        from_index: Index from which to truncate (inclusive).
                    Messages at and after this index are removed.

    Returns:
        Number of messages removed if successful.
        None if session was not found.

    Raises:
        ValueError: If from_index is negative.
    """
    if not session_store.exists(session_id):
        return None

    session = session_store.get(session_id)

    removed_count = session.truncate_messages(from_index)

    # Save updated session
    session_store.save(session)

    return removed_count
