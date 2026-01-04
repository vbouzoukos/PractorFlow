"""
Common helpers for LLM runner tests.

This module provides runner-specific test infrastructure used across
llama_cpp_runner and transformers_runner test files.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from practorflow.llm.pool.model_handle import ModelHandle


def create_mock_model_handle(
    backend: str = "llama_cpp",
    model_name: str = "test-model",
    max_context_length: int = 4096,
    config_hash: str = "abc123",
    chat_template: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    """
    Create a mock ModelHandle for testing.
    
    Args:
        backend: Backend type ("llama_cpp" or "transformers").
        model_name: Model name for the config.
        max_context_length: Maximum context length.
        config_hash: Configuration hash string.
        chat_template: Optional chat template string.
        metadata: Optional model metadata dict.
    
    Returns:
        MagicMock configured as a ModelHandle.
    """
    handle = MagicMock(spec=ModelHandle)
    handle.backend = backend
    handle.is_llama_cpp = backend == "llama_cpp"
    handle.is_transformers = backend == "transformers"
    handle.max_context_length = max_context_length
    handle.config_hash = config_hash
    
    handle.config = MagicMock()
    handle.config.model_name = model_name
    handle.config.backend = backend
    handle.config.device = "cpu"
    handle.config.dtype = "float16"
    handle.config.temperature = 0.7
    handle.config.top_p = 0.9
    handle.config.max_new_tokens = 512
    handle.config.max_search_results = 5
    handle.config.stop_tokens = None
    
    handle.model = create_mock_llama_model(
        chat_template=chat_template,
        metadata=metadata,
    )
    handle.tokenizer = None
    
    handle.get_chat_template = MagicMock(return_value=chat_template)
    
    return handle


def create_mock_llama_model(
    chat_template: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    """
    Create a mock llama-cpp-python Llama model.
    
    Args:
        chat_template: Optional chat template for metadata.
        metadata: Optional metadata dict. If not provided and chat_template
                  is given, metadata will be created with the template.
    
    Returns:
        MagicMock configured as a Llama model.
    """
    model = MagicMock()
    
    if metadata is None:
        metadata = {}
    if chat_template and "tokenizer.chat_template" not in metadata:
        metadata["tokenizer.chat_template"] = chat_template
    
    model.metadata = metadata
    model.create_chat_completion = MagicMock()
    model.n_ctx = MagicMock(return_value=4096)
    
    return model


def create_llama_completion_response(
    content: str = "This is the response.",
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a mock llama.cpp completion response.
    
    Args:
        content: The response text content.
        finish_reason: Reason for completion (e.g., "stop", "length").
        tool_calls: Optional list of tool calls in the response.
    
    Returns:
        Dict structured like llama.cpp create_chat_completion response.
    """
    message = {"role": "assistant", "content": content}
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def create_llama_stream_chunks(
    text_chunks: List[str],
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Create a list of mock llama.cpp streaming chunks.
    
    Args:
        text_chunks: List of text fragments to yield.
        finish_reason: Reason for completion on final chunk.
        tool_calls: Optional tool calls to include (spread across chunks).
    
    Returns:
        List of dicts structured like llama.cpp streaming chunks.
    """
    chunks = []
    
    for i, text in enumerate(text_chunks):
        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(chunk)
    
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_chunk = {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": tc.get("id", f"call_{i}"),
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": tc.get("function", {}).get("arguments", ""),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append(tool_chunk)
    
    final_chunk = {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    chunks.append(final_chunk)
    
    return chunks


def create_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a tool call dict in llama.cpp response format.
    
    Args:
        tool_name: Name of the tool/function.
        arguments: Dict of arguments to pass to the tool.
        call_id: Optional call ID (generated if not provided).
    
    Returns:
        Dict structured like a tool call in llama.cpp response.
    """
    import json
    
    return {
        "id": call_id or f"call_{tool_name}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    }


def create_incremental_tool_call_stream_chunks(
    tool_name: str,
    arguments: Dict[str, Any],
    call_id: str = "call_1",
    include_id_in_first_chunk: bool = True,
    finish_reason: str = "tool_calls",
) -> List[Dict[str, Any]]:
    """
    Create streaming chunks that incrementally build a tool call.
    
    This simulates how llama.cpp streams tool calls with name and arguments
    split across multiple chunks.
    
    Args:
        tool_name: Name of the tool/function.
        arguments: Dict of arguments for the tool.
        call_id: Tool call ID.
        include_id_in_first_chunk: Whether to include id in first chunk.
        finish_reason: Finish reason for final chunk.
    
    Returns:
        List of streaming chunks that build up the tool call.
    """
    import json
    
    args_json = json.dumps(arguments)
    mid_point = len(args_json) // 2
    
    chunks = []
    
    first_delta = {
        "tool_calls": [
            {
                "index": 0,
                "function": {"name": tool_name[:3], "arguments": ""},
            }
        ]
    }
    if include_id_in_first_chunk:
        first_delta["tool_calls"][0]["id"] = call_id
    
    chunks.append({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{"index": 0, "delta": first_delta, "finish_reason": None}],
    })
    
    chunks.append({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": tool_name[3:],
                                "arguments": args_json[:mid_point],
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    })
    
    chunks.append({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": args_json[mid_point:]},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    })
    
    chunks.append({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })
    
    return chunks