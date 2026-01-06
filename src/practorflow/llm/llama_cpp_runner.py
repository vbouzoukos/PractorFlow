import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List, AsyncIterator, Tuple

from practorflow.llm.base.llm_runner import LLMRunner, StreamChunk
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "llama_cpp_runner", level=appConfiguration.LoggerConfiguration.RunnerLevel
)


class LlamaCppRunner(LLMRunner):
    """Async runner for GGUF models using llama-cpp-python."""

    def __init__(
        self, handle: ModelHandle, knowledge_store: Optional[KnowledgeStore] = None
    ):
        if not handle.is_llama_cpp:
            raise ValueError(f"Expected llama_cpp handle, got {handle.backend}")

        super().__init__(handle, knowledge_store)
        self._thinking_model = self._chat_template_starts_with_think()
        logger.info(
            f"[LlamaCppRunner] Initialized with pooled model: {self.model_name}"
            f" (thinking_model={self._thinking_model})"
        )

    def _chat_template_starts_with_think(self) -> bool:
        """
        Detect if the chat template prepends <think> in the generation prompt.
        
        This checks the chat template for patterns that indicate the model
        starts its response with a <think> tag (e.g., DeepSeek-R1, QwQ models).
        
        Returns:
            True if the chat template prepends <think> to assistant responses.
        """
        try:
            metadata = self.model.metadata if hasattr(self.model, 'metadata') else {}
            chat_template = metadata.get("tokenizer.chat_template", "") if metadata else ""
            
            if not chat_template:
                return False
            
            template_str = str(chat_template)
            
            # Pattern 1: Look for generation_prompt that ends with <think>
            # Common pattern: {%- if add_generation_prompt %}...<think>{%- endif %}
            gen_prompt_think_pattern = re.compile(
                r'add_generation_prompt.*?<think>\s*\{%-?\s*endif\s*-?%\}',
                re.IGNORECASE | re.DOTALL
            )
            if gen_prompt_think_pattern.search(template_str):
                logger.info("[LlamaCppRunner] Detected thinking model (generation_prompt ends with <think>)")
                return True
            
            # Pattern 2: Assistant prefix contains <think> within 50 chars
            # e.g., assistant header like "<|assistant|>\n<think>"
            for match in re.finditer(r'assistant', template_str, re.IGNORECASE):
                start_pos = match.end()
                end_pos = min(start_pos + 50, len(template_str))
                snippet = template_str[start_pos:end_pos]
                if '<think>' in snippet:
                    logger.info("[LlamaCppRunner] Detected thinking model (assistant prefix contains <think>)")
                    return True
            
            # Pattern 3: Explicit think tag at end of template
            if template_str.rstrip().endswith('<think>'):
                logger.info("[LlamaCppRunner] Detected thinking model (template ends with <think>)")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"[LlamaCppRunner] Error detecting thinking model: {e}")
            return False

    def supports_function_calling(self) -> bool:
        """
        Check if the llama.cpp model supports native function calling.
        
        Returns:
            True if model supports native function calling, False otherwise
        """
        try:
            metadata = self.model.metadata if hasattr(self.model, 'metadata') else {}
            
            if metadata:
                chat_template = metadata.get("tokenizer.chat_template", "")
                if chat_template:
                    template_lower = str(chat_template).lower()
                    if any(keyword in template_lower for keyword in [
                        'tool', 'function', '<tool_call>', '<function_call>',
                        'tools', 'functions', 'tool_use', 'function_use'
                    ]):
                        logger.info(f"[LlamaCppRunner] Model supports function calling (detected in chat template)")
                        return True
            
            logger.info(f"[LlamaCppRunner] Model does NOT support native function calling")
            return False
            
        except Exception as e:
            logger.warning(f"[LlamaCppRunner] Error detecting function calling support: {e}")
            return False

    def _parse_thinking_response(self, text: str) -> Tuple[Optional[str], str]:
        """
        Parse response text to extract thinking/reasoning traces.
        
        Handles models like Nemotron 3 that use <think>...</think> tags
        for chain-of-thought reasoning.
        
        Args:
            text: Raw response text from the model
            
        Returns:
            Tuple of (thinking_content, reply_content):
                - thinking_content: Content inside <think> tags, or None if not present
                - reply_content: Content outside <think> tags (the actual response)
        """
        if not text:
            return None, ""
        
        # Pattern to match <think>...</think> blocks (handles multiline)
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        
        # Find all thinking blocks
        think_matches = think_pattern.findall(text)
        
        if not think_matches:
            # No thinking tags found, return original text as reply
            return None, text.strip()
        
        # Combine all thinking content
        thinking_content = "\n".join(match.strip() for match in think_matches if match.strip())
        
        # Remove thinking blocks from text to get the reply
        reply_content = think_pattern.sub('', text).strip()
        
        # Handle empty thinking (e.g., <think></think>)
        if not thinking_content:
            thinking_content = None
        
        logger.debug(f"[LlamaCppRunner] Parsed thinking: {len(thinking_content) if thinking_content else 0} chars, reply: {len(reply_content)} chars")
        
        return thinking_content, reply_content

    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from llama.cpp response.
        
        Args:
            response: Response from create_chat_completion
            
        Returns:
            Text content from the response
        """
        choices = response.get("choices", [])
        if not choices:
            return ""
        
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    def _convert_tools_to_llama_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tool definitions to llama.cpp native format.
        
        Args:
            tools: List of tool definitions in OpenAI/Pydantic AI format
            
        Returns:
            List of tools in llama.cpp format
        """
        llama_tools = []
        for tool in tools:
            if "function" in tool:
                # Already in OpenAI format
                llama_tools.append(tool)
            elif "name" in tool and "description" in tool:
                # Pydantic AI ToolDefinition format
                llama_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("parameters", tool.get("parameters_json_schema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }))
                    }
                }
                llama_tools.append(llama_tool)
            else:
                logger.warning(f"[LlamaCppRunner] Unknown tool format: {tool}")
        return llama_tools

    def _extract_tool_calls_from_response(
        self, 
        response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract tool calls from llama.cpp native response.
        
        Args:
            response: Response from create_chat_completion
            
        Returns:
            List of tool call dicts with tool_name, args, tool_call_id
        """
        tool_calls = []
        choices = response.get("choices", [])
        
        if not choices:
            return tool_calls
        
        message = choices[0].get("message", {})
        native_tool_calls = message.get("tool_calls", [])
        
        for tc in native_tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            
            # Parse arguments
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
                logger.warning(f"[LlamaCppRunner] Failed to parse tool arguments: {args_str}")
            
            tool_calls.append({
                "tool_name": tool_name,
                "args": args,
                "tool_call_id": tc.get("id", f"call_{len(tool_calls)}")
            })
        
        return tool_calls

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages array for create_chat_completion API."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")

        chat_messages = []
        system_parts = []

        # Add custom instructions first
        if instructions:
            system_parts.append(instructions)

        # Add context with clear instructions for the model to use it
        if context:
            system_parts.append(
                "You have access to the following REFERENCE DOCUMENTS. "
                "Use ONLY this information to answer the user's question. "
                "If the answer is in the documents, provide it. "
                "Quote or reference specific parts when relevant."
            )
            system_parts.append(f"REFERENCE DOCUMENTS:\n{context}\nEND OF DOCUMENTS.")

        if system_parts:
            chat_messages.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )

        if messages is not None:
            for msg in messages:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            # When context is provided, remind the model to use it
            user_content = prompt
            if context:
                user_content = f"{prompt}\n\n(Answer based on the reference documents provided above.)"
            chat_messages.append({"role": "user", "content": user_content})

        return chat_messages

    def _generate_sync(
        self,
        chat_messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synchronous generation - runs in thread pool."""
        completion_kwargs = {
            "messages": chat_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_new_tokens,
        }

        if self.config.stop_tokens:
            completion_kwargs["stop"] = self.config.stop_tokens

        # Add native tools if provided
        if tools:
            llama_tools = self._convert_tools_to_llama_format(tools)
            if llama_tools:
                completion_kwargs["tools"] = llama_tools
                completion_kwargs["tool_choice"] = "auto"
                logger.debug(f"[LlamaCppRunner] Passing {len(llama_tools)} tools to native API")

        return self.model.create_chat_completion(**completion_kwargs)

    async def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Async generate with optional context from prior search() call.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions for native function calling
            
        Returns:
            Dictionary with reply, latency_seconds, and optional thinking/tool_calls/context
        """
        start_time = time.time()

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(f"[LlamaCppRunner] Using search context ({len(context)} chars)")

        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[LlamaCppRunner] Generating (async non-streaming)...")
        if tools:
            logger.info(f"[LlamaCppRunner] With {len(tools)} native tools")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, chat_messages, temp, tp, tools
        )

        latency = time.time() - start_time

        # Extract content and parse thinking
        raw_content = self._extract_content_from_response(result)
        
        # For thinking models, handle case where response starts mid-thinking
        # (template already added <think>, so content may start with thinking)
        if self._thinking_model and '</think>' in raw_content:
            # Split on </think> - content before is thinking, after is reply
            parts = raw_content.split('</think>', 1)
            thinking_part = parts[0].strip()
            reply_part = parts[1].strip() if len(parts) > 1 else ""
            # Remove leading <think> if present
            if thinking_part.startswith('<think>'):
                thinking_part = thinking_part[7:].strip()
            thinking = thinking_part if thinking_part else None
            reply = reply_part
        elif self._thinking_model and '</think>' not in raw_content:
            # No closing tag found - this is a normal model response
            thinking = None
            reply = raw_content.strip()
            # Remove leading <think> if present (from template)
            if reply.startswith('<think>'):
                reply = reply[7:].strip()
        else:
            thinking, reply = self._parse_thinking_response(raw_content)

        # Extract tool calls if present
        tool_calls = self._extract_tool_calls_from_response(result)

        response = {
            "reply": reply,
            "latency_seconds": latency,
        }

        # Include thinking if present
        if thinking:
            response["thinking"] = thinking
            logger.info(f"[LlamaCppRunner] Extracted thinking trace ({len(thinking)} chars)")

        if tool_calls:
            response["tool_calls"] = tool_calls
            logger.info(f"[LlamaCppRunner] Native tool calls extracted: {[tc['tool_name'] for tc in tool_calls]}")

        if context:
            response["context_used"] = context
            if context_metadata:
                response["search_metadata"] = context_metadata

        return response

    async def generate_stream(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Async streaming generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions for native function calling
            
        Yields:
            StreamChunk objects with text deltas and final metadata
        """
        start_time = time.time()

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(f"[LlamaCppRunner] Using search context ({len(context)} chars)")

        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[LlamaCppRunner] Generating (async streaming)...")
        if tools:
            logger.info(f"[LlamaCppRunner] With {len(tools)} native tools")

        # Detect if this is a thinking model (template prepends <think>)
        is_thinking_model = self._thinking_model

        # Create async queue to bridge sync generator to async
        queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue()

        def stream_in_thread():
            """Run streaming generation in thread, put chunks in queue."""
            try:
                completion_kwargs = {
                    "messages": chat_messages,
                    "temperature": temp,
                    "top_p": tp,
                    "max_tokens": self.max_new_tokens,
                    "stream": True,
                }

                if self.config.stop_tokens:
                    completion_kwargs["stop"] = self.config.stop_tokens

                # Add native tools if provided
                if tools:
                    llama_tools = self._convert_tools_to_llama_format(tools)
                    if llama_tools:
                        completion_kwargs["tools"] = llama_tools
                        completion_kwargs["tool_choice"] = "auto"

                finish_reason = None
                accumulated_tool_calls = []
                accumulated_text = ""  # Track full response for thinking extraction
                accumulated_thinking = ""  # Track thinking content separately
                # For thinking models, start in thinking mode
                in_thinking = is_thinking_model
                saw_think_close = False  # Track if we ever saw </think>
                pending_text = ""  # Buffer for handling partial tags

                for chunk in self.model.create_chat_completion(**completion_kwargs):
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    chunk_finish_reason = choice.get("finish_reason")
                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason

                    # Handle streaming tool calls
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(accumulated_tool_calls) <= idx:
                                accumulated_tool_calls.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            if "id" in tc:
                                accumulated_tool_calls[idx]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    accumulated_tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    accumulated_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

                    if content:
                        # Accumulate full text for reference
                        accumulated_text += content
                        
                        # Add to pending buffer for tag processing
                        pending_text += content
                        
                        # Process pending text for <think> tags
                        text_to_yield = ""
                        while pending_text:
                            if not in_thinking:
                                # Look for <think> tag
                                think_start = pending_text.find("<think>")
                                if think_start == -1:
                                    # No <think> tag, check if we might have partial tag at end
                                    # Keep last 6 chars in case of partial "<think"
                                    if len(pending_text) > 6 and pending_text[-6:].startswith("<"):
                                        text_to_yield += pending_text[:-6]
                                        pending_text = pending_text[-6:]
                                    else:
                                        text_to_yield += pending_text
                                        pending_text = ""
                                elif think_start > 0:
                                    # Text before <think> tag
                                    text_to_yield += pending_text[:think_start]
                                    pending_text = pending_text[think_start:]
                                else:
                                    # Starts with <think>
                                    in_thinking = True
                                    pending_text = pending_text[7:]  # Skip "<think>"
                            else:
                                # Inside thinking block, look for </think>
                                think_end = pending_text.find("</think>")
                                if think_end == -1:
                                    # No </think> yet, accumulate thinking content
                                    # But keep buffer in case of partial "</think"
                                    if len(pending_text) > 7 and pending_text[-7:].startswith("<"):
                                        accumulated_thinking += pending_text[:-7]
                                        pending_text = pending_text[-7:]
                                    else:
                                        accumulated_thinking += pending_text
                                        pending_text = ""
                                    break
                                else:
                                    # Found </think>, capture thinking content before it
                                    accumulated_thinking += pending_text[:think_end]
                                    saw_think_close = True
                                    in_thinking = False
                                    pending_text = pending_text[think_end + 8:]  # Skip "</think>"
                        
                        if text_to_yield:
                            # Put chunk in queue (blocking call from thread)
                            asyncio.run_coroutine_threadsafe(
                                queue.put(StreamChunk(text=text_to_yield, finished=False)), loop
                            ).result()

                # Handle remaining pending_text at end of stream
                if pending_text or accumulated_thinking:
                    if in_thinking and not saw_think_close and is_thinking_model:
                        # Thinking model but no </think> found - treat as normal model
                        # Yield all accumulated content as regular text
                        all_text = accumulated_thinking + pending_text
                        if all_text:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(StreamChunk(text=all_text, finished=False)), loop
                            ).result()
                        # Clear thinking since this is normal output
                        accumulated_thinking = ""
                    elif in_thinking and not is_thinking_model:
                        # Non-thinking model with incomplete thinking block
                        # Yield the pending content as regular text
                        all_text = accumulated_thinking + pending_text
                        if all_text:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(StreamChunk(text=all_text, finished=False)), loop
                            ).result()
                        accumulated_thinking = ""

                # Process accumulated tool calls
                tool_calls_result = []
                for tc in accumulated_tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    if tool_name:
                        tool_calls_result.append({
                            "tool_name": tool_name,
                            "args": args,
                            "tool_call_id": tc.get("id", f"call_{len(tool_calls_result)}")
                        })

                # Determine thinking result for final metadata
                thinking_result = None
                if saw_think_close and accumulated_thinking.strip():
                    # We found proper thinking blocks
                    thinking_result = accumulated_thinking.strip()
                elif not is_thinking_model and accumulated_text:
                    # Non-thinking model, parse from accumulated text
                    thinking_result, _ = self._parse_thinking_response(accumulated_text)

                # Build final search_metadata including thinking if present
                final_metadata = context_metadata.copy() if context_metadata else {}
                if thinking_result:
                    final_metadata["thinking"] = thinking_result

                # Put final chunk
                latency = time.time() - start_time
                final_chunk = StreamChunk(
                    text="",
                    finished=True,
                    finish_reason=finish_reason,
                    latency_seconds=latency,
                    context_used=context,
                    search_metadata=final_metadata if final_metadata else context_metadata,
                )
                
                # Attach tool calls to final chunk if present
                if tool_calls_result:
                    final_chunk.tool_calls = tool_calls_result
                    
                asyncio.run_coroutine_threadsafe(queue.put(final_chunk), loop).result()

            except Exception as e:
                # Put error chunk
                error_chunk = StreamChunk(
                    text="", finished=True, finish_reason=f"error: {str(e)}"
                )
                asyncio.run_coroutine_threadsafe(queue.put(error_chunk), loop).result()
            finally:
                # Signal end of stream
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Start generation in thread
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, stream_in_thread)

        # Yield chunks from queue
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk