import asyncio
import re
import time
from threading import Thread
from typing import Dict, Any, Optional, List, AsyncIterator, Tuple

import torch
from transformers import TextIteratorStreamer

from practorflow.llm.base.llm_runner import LLMRunner, StreamChunk
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "transformers_runner", level=appConfiguration.LoggerConfiguration.RunnerLevel
)


class TransformersRunner(LLMRunner):
    """Async runner for HuggingFace Transformers models with optimizations."""

    def __init__(
        self, handle: ModelHandle, knowledge_store: Optional[KnowledgeStore] = None
    ):
        if not handle.is_transformers:
            raise ValueError(f"Expected transformers handle, got {handle.backend}")

        super().__init__(handle, knowledge_store)

        if hasattr(self.model, "device"):
            self._device = self.model.device
        else:
            self._device = next(self.model.parameters()).device

        # Pre-compute generation defaults to avoid repeated attribute access
        self._pad_token_id = self.tokenizer.pad_token_id
        self._eos_token_id = self.tokenizer.eos_token_id

        # Detect if this is a thinking model
        self._thinking_model = self._chat_template_starts_with_think()

        logger.info(
            f"[TransformersRunner] Initialized with pooled model: {self.model_name}"
            f" (thinking_model={self._thinking_model})"
        )
        logger.info(f"[TransformersRunner] Device: {self._device}")

    def _chat_template_starts_with_think(self) -> bool:
        """
        Detect if the chat template prepends <think> in the generation prompt.
        
        Returns:
            True if the chat template prepends <think> to assistant responses.
        """
        try:
            chat_template = None
            if self.tokenizer and hasattr(self.tokenizer, 'chat_template'):
                chat_template = self.tokenizer.chat_template
            
            if not chat_template:
                return False
            
            template_str = str(chat_template)
            
            # Pattern 1: Look for generation_prompt that ends with <think>
            gen_prompt_think_pattern = re.compile(
                r'add_generation_prompt.*?<think>\s*\{%-?\s*endif\s*-?%\}',
                re.IGNORECASE | re.DOTALL
            )
            if gen_prompt_think_pattern.search(template_str):
                logger.info("[TransformersRunner] Detected thinking model (generation_prompt ends with <think>)")
                return True
            
            # Pattern 2: Assistant prefix contains <think> within 50 chars
            for match in re.finditer(r'assistant', template_str, re.IGNORECASE):
                start_pos = match.end()
                end_pos = min(start_pos + 50, len(template_str))
                snippet = template_str[start_pos:end_pos]
                if '<think>' in snippet:
                    logger.info("[TransformersRunner] Detected thinking model (assistant prefix contains <think>)")
                    return True
            
            # Pattern 3: Explicit think tag at end of template
            if template_str.rstrip().endswith('<think>'):
                logger.info("[TransformersRunner] Detected thinking model (template ends with <think>)")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"[TransformersRunner] Error detecting thinking model: {e}")
            return False 

    def supports_function_calling(self) -> bool:
        """
        Check if the transformers model supports native function calling.
        
        Returns:
            True if model supports native function calling, False otherwise
        """
        try:
            if self.tokenizer and hasattr(self.tokenizer, "chat_template"):
                chat_template = self.tokenizer.chat_template
                if chat_template:
                    template_lower = str(chat_template).lower()
                    if any(
                        keyword in template_lower
                        for keyword in [
                            "tool",
                            "function",
                            "<tool_call>",
                            "<function_call>",
                            "tools",
                            "functions",
                            "tool_use",
                            "function_use",
                        ]
                    ):
                        logger.info(
                            f"[TransformersRunner] Model supports function calling (detected in chat template)"
                        )
                        return True

            if hasattr(self.model, "config"):
                config = self.model.config
                if hasattr(config, "to_dict"):
                    config_dict = config.to_dict()
                    if config_dict.get("supports_function_calling", False):
                        logger.info(
                            f"[TransformersRunner] Model supports function calling (config flag)"
                        )
                        return True

            logger.info(
                f"[TransformersRunner] Model does NOT support native function calling"
            )
            return False

        except Exception as e:
            logger.warning(
                f"[TransformersRunner] Error detecting function calling support: {e}"
            )
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
        
        logger.debug(f"[TransformersRunner] Parsed thinking: {len(thinking_content) if thinking_content else 0} chars, reply: {len(reply_content)} chars")
        
        return thinking_content, reply_content

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages for the model."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")

        chat_messages = []
        system_parts = []

        if instructions:
            system_parts.append(instructions)

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
            user_content = prompt
            if context:
                user_content = f"{prompt}\n\n(Answer based on the reference documents provided above.)"
            chat_messages.append({"role": "user", "content": user_content})

        return chat_messages

    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback message formatting for tokenizers without chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _prepare_inputs(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> tuple:
        """Prepare tokenized inputs for generation."""
        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = self._format_messages_fallback(chat_messages)

        # Tokenize and move to device in one step
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=(
                self.max_context_length - self.max_new_tokens
                if self.max_context_length
                else None
            ),
        ).to(self._device, non_blocking=True)

        input_length = inputs["input_ids"].shape[1]

        return inputs, input_length

    def _build_generation_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        temperature: float,
        top_p: float,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> Dict[str, Any]:
        """Build generation kwargs dict to avoid repetition."""
        do_sample = temperature > 0

        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pad_token_id,
            "eos_token_id": self._eos_token_id,
        }

        # Only include sampling params when do_sample=True
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        return gen_kwargs

    def _generate_sync(
        self,
        inputs: Dict[str, torch.Tensor],
        input_length: int,
        temperature: float,
        top_p: float,
    ) -> tuple:
        """Synchronous generation - runs in thread pool."""
        gen_kwargs = self._build_generation_kwargs(inputs, temperature, top_p)

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return response_text, len(generated_tokens)

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
            tools: Optional list of tool definitions (not used in transformers backend currently)

        Returns:
            Dictionary with reply, latency_seconds, usage, and optionally context info
        """
        start_time = time.perf_counter()

        if tools:
            logger.warning(
                "[TransformersRunner] Native tool calling not implemented for transformers backend, tools will be ignored"
            )

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(
                f"[TransformersRunner] Using search context ({len(context)} chars)"
            )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[TransformersRunner] Generating ...")

        inputs, input_length = self._prepare_inputs(
            messages, prompt, instructions, context
        )

        loop = asyncio.get_running_loop()
        response_text, completion_tokens = await loop.run_in_executor(
            None, self._generate_sync, inputs, input_length, temp, tp
        )

        latency = time.perf_counter() - start_time

        # Handle thinking model response
        if self._thinking_model and '</think>' in response_text:
            # Split on </think> - content before is thinking, after is reply
            parts = response_text.split('</think>', 1)
            thinking_part = parts[0].strip()
            reply_part = parts[1].strip() if len(parts) > 1 else ""
            # Remove leading <think> if present
            if thinking_part.startswith('<think>'):
                thinking_part = thinking_part[7:].strip()
            thinking = thinking_part if thinking_part else None
            reply = reply_part
        elif self._thinking_model and '</think>' not in response_text:
            # No closing tag found - this is a normal model response
            thinking = None
            reply = response_text.strip()
            # Remove leading <think> if present (from template)
            if reply.startswith('<think>'):
                reply = reply[7:].strip()
        else:
            # Parse thinking from response
            thinking, reply = self._parse_thinking_response(response_text)

        response = {
            "reply": reply,
            "latency_seconds": latency,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
        }

        # Include thinking if present
        if thinking:
            response["thinking"] = thinking
            logger.info(f"[TransformersRunner] Extracted thinking trace ({len(thinking)} chars)")

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
        Async streaming generation using TextIteratorStreamer.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions (not used in transformers backend currently)

        Yields:
            StreamChunk objects with text deltas and final metadata
        """
        start_time = time.perf_counter()

        if tools:
            logger.warning(
                "[TransformersRunner] Native tool calling not implemented for transformers backend, tools will be ignored"
            )

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(
                f"[TransformersRunner] Using search context ({len(context)} chars)"
            )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[TransformersRunner] Streaming ...")

        inputs, input_length = self._prepare_inputs(
            messages, prompt, instructions, context
        )

        # Create streamer with skip_prompt to avoid re-emitting input
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = self._build_generation_kwargs(inputs, temp, tp, streamer)

        # Track generated tokens via output length after generation
        generation_complete = asyncio.Event()
        generation_error: Optional[Exception] = None
        output_ids: Optional[torch.Tensor] = None

        def generate_in_thread():
            nonlocal generation_error, output_ids
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(**gen_kwargs)
                    output_ids = outputs[0]
            except Exception as e:
                generation_error = e
            finally:
                generation_complete.set()

        # Start generation thread
        thread = Thread(target=generate_in_thread, daemon=True)
        thread.start()

        # Stream tokens from streamer
        loop = asyncio.get_running_loop()

        # Detect if this is a thinking model
        is_thinking_model = self._thinking_model

        # Track state for thinking tag filtering
        accumulated_text = ""  # Track full response for thinking extraction
        accumulated_thinking = ""  # Track thinking content separately
        in_thinking = is_thinking_model  # Start in thinking mode for thinking models
        saw_think_close = False  # Track if we ever saw </think>
        pending_text = ""  # Buffer for handling partial tags

        def get_next_token():
            """Blocking call to get next token from streamer."""
            try:
                return next(iter(streamer))
            except StopIteration:
                return None

        while True:
            # Get next text chunk from streamer (blocking in thread pool)
            text = await loop.run_in_executor(None, get_next_token)

            if text is None:
                break

            if text:
                # Accumulate text for final thinking extraction
                accumulated_text += text
                
                # Add to pending buffer for tag processing
                pending_text += text
                
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
                    yield StreamChunk(text=text_to_yield, finished=False)

        # Handle remaining pending_text at end of stream
        if pending_text or accumulated_thinking:
            if in_thinking and not saw_think_close and is_thinking_model:
                # Thinking model but no </think> found - treat as normal model
                # Yield all accumulated content as regular text
                all_text = accumulated_thinking + pending_text
                if all_text:
                    yield StreamChunk(text=all_text, finished=False)
                # Clear thinking since this is normal output
                accumulated_thinking = ""
            elif in_thinking and not is_thinking_model:
                # Non-thinking model with incomplete thinking block
                # Yield the pending content as regular text
                all_text = accumulated_thinking + pending_text
                if all_text:
                    yield StreamChunk(text=all_text, finished=False)
                accumulated_thinking = ""

        # Wait for generation to complete
        await generation_complete.wait()

        # Calculate final stats
        latency = time.perf_counter() - start_time

        completion_tokens = 0
        if output_ids is not None:
            completion_tokens = len(output_ids) - input_length

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

        # Yield final chunk with metadata
        final_chunk = StreamChunk(
            text="",
            finished=True,
            finish_reason="error" if generation_error else "stop",
            latency_seconds=latency,
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
            context_used=context,
            search_metadata=final_metadata if final_metadata else context_metadata,
        )

        if generation_error:
            logger.error(f"[TransformersRunner] Generation error: {generation_error}")
            final_chunk.finish_reason = f"error: {str(generation_error)}"

        yield final_chunk