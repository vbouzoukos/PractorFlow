"""
Common test helpers for PractorFlow tests.

Provides reusable utilities for testing async streaming, mocking,
and other common test patterns across the test suite.
"""

from typing import List, Union

from practorflow.llm import StreamChunk


class MockStreamText:
    """
    Mock async iterator for streaming responses.
    
    Used for testing async streaming functionality without requiring
    actual LLM model execution. Implements the async iterator protocol
    to work with 'async for' loops.
    
    Accepts either StreamChunk objects or plain strings. When strings
    are provided, they are yielded as-is (for mocking stream_text()).
    When StreamChunk objects are provided, they are yielded as-is
    (for mocking generate_stream()).
    
    Example with StreamChunk (for generate_stream):
        chunks = [
            StreamChunk(text="Hello", finished=False),
            StreamChunk(text=" world", finished=False),
            StreamChunk(text="", finished=True, finish_reason="stop"),
        ]
        mock_stream = MockStreamText(chunks)
        
        async for chunk in mock_stream:
            print(chunk.text)
    
    Example with strings (for stream_text):
        texts = ["Hello", " world", "!"]
        mock_stream = MockStreamText(texts)
        
        async for text in mock_stream:
            print(text)
    """

    def __init__(self, items: Union[List[StreamChunk], List[str]]):
        """
        Initialize the mock stream.
        
        Args:
            items: List of StreamChunk objects or strings to yield in sequence.
        """
        self.items = items
        self.index = 0

    def __aiter__(self):
        """Return self as the async iterator."""
        return self

    async def __anext__(self):
        """
        Get the next item in the sequence.
        
        Returns:
            Next StreamChunk object or string.
            
        Raises:
            StopAsyncIteration: When all items have been yielded.
        """
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        raise StopAsyncIteration