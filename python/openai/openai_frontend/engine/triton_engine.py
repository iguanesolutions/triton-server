# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, List, Optional

import tritonserver
import re
from engine.engine import LLMEngine
from engine.utils.tokenizer import get_tokenizer
from engine.utils.triton import (
    _create_trtllm_inference_request,
    _create_vllm_inference_request,
    _get_output,
    _validate_triton_responses_non_streaming,
)
from schemas.openai import (
    CompletionUsage,
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionToolChoiceOption1,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateChatCompletionStreamOptions,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    Function1,
    Function2,
    Model,
    ObjectType
)


# TODO: Improve type hints
@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model
    # Tokenizers used for chat templates
    tokenizer: Optional[Any]
    # Time that model was loaded by Triton
    create_time: int
    # Conversion format between OpenAI and Triton requests
    request_converter: Callable


class TritonLLMEngine(LLMEngine):
    def __init__(
        self, server: tritonserver.Server, tokenizer: str, backend: Optional[str] = None
    ):
        # Assume an already configured and started server
        self.server = server
        self.tokenizer = self._get_tokenizer(tokenizer)
        # TODO: Reconsider name of "backend" vs. something like "request_format"
        self.backend = backend

        # NOTE: Creation time and model metadata will be static at startup for
        # now, and won't account for dynamically loading/unloading models.
        self.create_time = int(time.time())
        self.model_metadata = self._get_model_metadata()

    def ready(self) -> bool:
        return self.server.ready()

    def metrics(self) -> str:
        return self.server.metrics()

    def models(self) -> List[Model]:
        models = []
        for metadata in self.model_metadata.values():
            models.append(
                Model(
                    id=metadata.name,
                    created=metadata.create_time,
                    object=ObjectType.model,
                    owned_by="Triton Inference Server",
                ),
            )

        return models

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | AsyncIterator[str]:
        metadata = self.model_metadata.get(request.model)
        self._validate_chat_request(request, metadata)

        conversation = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]

        if request.tools is not None and request.tool_choice.root != ChatCompletionToolChoiceOption1.none:
            tools = []
            for chatcompletiontool in request.tools:
                tool_def = chatcompletiontool.model_dump(exclude={'type'})
                tool_def['type'] = "function"
                tools.append(tool_def)
        else:
            tools = None
        add_generation_prompt = True
        prompt = metadata.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools
        )

        if request.seed is None:
            request.seed = random.randint(-9223372036854775808, 9223372036854775807)

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(metadata.model, prompt, request)
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        default_role = "assistant"
        role = self._get_first_response_role(
            conversation, add_generation_prompt, default_role
        )

        if request.stream:
            return self._streaming_chat_iterator(
                request_id, created, request.model, role, responses, prompt, request.stream_options,
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)

        response = responses[0]
        text = _get_output(response)

        # Parse tool calls if any
        tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        tool_call_tuples = tool_call_regex.findall(text)
        raw_function_calls = [
            json.loads(match[0] if match[0] else match[1]) # TODO why match[0] or match[1]
            for match in tool_call_tuples
        ]
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid1()}",
                type="function",
                function=Function1(
                    name=function_call["name"],
                    arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                ),
            )
            for function_call in raw_function_calls
        ]

        # Compute usage before text might be wiped out
        prompt_tokens = metadata.tokenizer.tokenize(prompt)
        completion_tokens  = metadata.tokenizer.tokenize(text)

        if tool_calls:
            tool_calls = ChatCompletionMessageToolCalls(root=tool_calls)
            text = ""
            finish_reason = ChatCompletionFinishReason.tool_calls
        else:
            tool_calls = None
            finish_reason = ChatCompletionFinishReason.stop

        return CreateChatCompletionResponse(
            id=request_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=text, role=role, function_call=None, tool_calls=tool_calls,
                    ),
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            created=created,
            model=request.model,
            system_fingerprint=None,
            object=ObjectType.chat_completion,
            usage=CompletionUsage(
                prompt_tokens = len(prompt_tokens),
                completion_tokens = len(completion_tokens),
                total_tokens = len(prompt_tokens) + len(completion_tokens),
            ),
        )

    async def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | AsyncIterator[str]:
        # Validate request and convert to Triton format
        metadata = self.model_metadata.get(request.model)
        self._validate_completion_request(request, metadata)

        if request.seed is None:
            request.seed = random.randint(-9223372036854775808, 9223372036854775807)

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(metadata.model, request.prompt, request)
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        if request.stream:
            return self._streaming_completion_iterator(
                request_id, created, metadata.name, responses, request.prompt, request.stream_options
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        choice = Choice(
            finish_reason=FinishReason.stop,
            index=0,
            logprobs=None,
            text=text,
        )

        # Compute usage
        prompt_tokens = metadata.tokenizer.tokenize(request.prompt)
        completion_tokens  = metadata.tokenizer.tokenize(text)

        return CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=metadata.name,
            usage=CompletionUsage(
                prompt_tokens = len(prompt_tokens),
                completion_tokens = len(completion_tokens),
                total_tokens = len(prompt_tokens) + len(completion_tokens),
            ),
        )

    # TODO: This behavior should be tested further
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    # TODO: Expose explicit flag to catch edge cases
    def _determine_request_converter(self, backend: str):
        # Allow manual override of backend request format if provided by user
        if self.backend:
            backend = self.backend

        # Request conversion from OpenAI format to backend-specific format
        if backend == "vllm":
            return _create_vllm_inference_request

        # Use TRT-LLM format as default for everything else. This could be
        # an ensemble, a python or BLS model, a TRT-LLM backend model, etc.
        return _create_trtllm_inference_request

    def _get_tokenizer(self, tokenizer_name: str):
        tokenizer = None
        if tokenizer_name:
            tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def _get_model_metadata(self) -> Dict[str, TritonModelMetadata]:
        # One tokenizer and creation time shared for all loaded models for now.
        model_metadata = {}

        # Read all triton models and store the necessary metadata for each
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            backend = model.config()["backend"]
            # Explicitly handle ensembles to avoid any runtime validation errors
            if not backend and model.config()["platform"] == "ensemble":
                backend = "ensemble"
            print(f"Found model: {name=}, {backend=}")

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=self.tokenizer,
                create_time=self.create_time,
                request_converter=self._determine_request_converter(backend),
            )
            model_metadata[name] = metadata

        return model_metadata

    def _get_streaming_chat_response_chunk(
        self,
        choice: ChatCompletionStreamingResponseChoice,
        request_id: str,
        created: int,
        model: str,
    ) -> CreateChatCompletionStreamResponse:
        return CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
            usage=None,
        )

    def _get_first_streaming_chat_response(
        self, request_id: str, created: int, model: str, role: str
    ) -> CreateChatCompletionStreamResponse:
        # First chunk has no content and sets the role
        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=role, content="", function_call=None
            ),
            logprobs=None,
            finish_reason=None,
        )
        chunk = self._get_streaming_chat_response_chunk(
            choice, request_id, created, model
        )
        return chunk

    def _get_nth_streaming_chat_response(
        self,
        request_id: str,
        created: int,
        model: str,
        text: str,
        tool_call_index: int,
        buf: str,
        final: bool,
    ) -> CreateChatCompletionStreamResponse:
        finish_reason = None
        if final:
            if tool_call_index > -1:
                finish_reason = ChatCompletionFinishReason.tool_calls
            else:
                finish_reason = ChatCompletionFinishReason.stop

        if tool_call_index > -1 and buf != "":
            tc = json.loads(buf)
            delta = ChatCompletionStreamResponseDelta(
                role=None, content="", function_call=None, tool_calls=[ChatCompletionMessageToolCallChunk(
                    id=f"call_{uuid.uuid1()}", type="function", index=tool_call_index, function=Function2(
                        name=tc["name"], arguments=json.dumps(tc["arguments"])
                    )
                )]
            )
        else:
            delta = ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            )

        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=delta,
            logprobs=None,
            finish_reason=finish_reason,
        )

        chunk = self._get_streaming_chat_response_chunk(
            choice, request_id, created, model
        )
        return chunk

    async def _streaming_chat_iterator(
        self,
        request_id: str,
        created: int,
        model: str,
        role: str,
        responses: AsyncIterable,
        prompt: str, # for usage
        stream_options: CreateChatCompletionStreamOptions,
    ) -> AsyncIterator[str]:
        chunk = self._get_first_streaming_chat_response(
            request_id, created, model, role
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
        consolidated_response: str = chunk.choices[0].delta.content # empty start

        tool_call_start_token = "<tool_call>"
        tool_call_end_token = "</tool_call>"
        tool_call_index = -1
        tool_call_buf = ""
        parsing_tool_call = False
        last_text = ""

        async for response in responses:
            text = _get_output(response)

            # we only need consolidated_response for usage
            if stream_options is not None and stream_options.include_usage:
                consolidated_response += text

            # skip extra newline after tool call end token
            if last_text == tool_call_end_token and text == "\n":
                continue
            # save last text for newline skipping logic
            last_text = text

            # check if we are parsing a tool call
            if text == tool_call_start_token:
                tool_call_index += 1
                parsing_tool_call = True
                continue
            elif text == tool_call_end_token:
                # write tool call if we reach end token
                chunk = self._get_nth_streaming_chat_response(
                    request_id, created, model, text, tool_call_index, tool_call_buf, response.final
                )
                tool_call_buf = ""
                parsing_tool_call = False
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                continue

            # tool call buffer while we are parsing a tool call
            if parsing_tool_call:
                tool_call_buf += text
                continue

            chunk = self._get_nth_streaming_chat_response(
                request_id, created, model, text, tool_call_index, "", response.final
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        # Compute usage after the loop
        if stream_options is not None and stream_options.include_usage:
            metadata = self.model_metadata.get(model)
            prompt_tokens = metadata.tokenizer.tokenize(prompt)
            completion_tokens  = metadata.tokenizer.tokenize(consolidated_response)
            chunk = CreateChatCompletionStreamResponse(
                id=request_id,
                choices=[],
                created=created,
                model=model,
                system_fingerprint=None,
                object=ObjectType.chat_completion_chunk,
                usage=CompletionUsage(
                    prompt_tokens = len(prompt_tokens),
                    completion_tokens = len(completion_tokens),
                    total_tokens = len(prompt_tokens) + len(completion_tokens),
                ),
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_chat_request(
        self, request: CreateChatCompletionRequest, metadata: TritonModelMetadata
    ):
        """
        Validates a chat request to align with currently supported features.
        """

        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.tokenizer:
            raise Exception("Unknown tokenizer")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        # Reject unsupported features if requested
        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs:
            raise Exception("logit bias and log probs not currently supported")

    async def _streaming_completion_iterator(
        self, request_id: str, created: int, model: str, responses: AsyncIterable,
        prompt: str, stream_options: Optional[CreateChatCompletionStreamOptions],
    ) -> AsyncIterator[str]:
        consolidated_response: str = ""
        async for response in responses:
            text = _get_output(response)
            choice = Choice(
                finish_reason=FinishReason.stop if response.final else None,
                index=0,
                logprobs=None,
                text=text,
            )
            chunk = CreateCompletionResponse(
                id=request_id,
                choices=[choice],
                system_fingerprint=None,
                object=ObjectType.text_completion,
                created=created,
                model=model,
                usage=None,
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            if stream_options is not None and stream_options.include_usage:
                consolidated_response += text

        if stream_options is not None and stream_options.include_usage:
            metadata = self.model_metadata.get(model)
            prompt_tokens = metadata.tokenizer.tokenize(prompt)
            completion_tokens  = metadata.tokenizer.tokenize(consolidated_response)
            chunk = CreateCompletionResponse(
                id=request_id,
                choices=[],
                system_fingerprint=None,
                object=ObjectType.text_completion,
                created=created,
                model=model,
                usage=CompletionUsage(
                    prompt_tokens = len(prompt_tokens),
                    completion_tokens = len(completion_tokens),
                    total_tokens = len(prompt_tokens) + len(completion_tokens),
                ),
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_completion_request(
        self, request: CreateCompletionRequest, metadata: TritonModelMetadata
    ):
        """
        Validates a completions request to align with currently supported features.
        """
        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        # Reject unsupported features if requested
        if request.suffix is not None:
            raise Exception("suffix is not currently supported")

        if not request.prompt:
            raise Exception("prompt must be non-empty")

        # Currently only support single string as input
        if not isinstance(request.prompt, str):
            raise Exception("only single string input is supported")

        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.best_of and request.best_of > 1:
            raise Exception(
                f"Received best_of={request.best_of}, but only single choice (best_of=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs is not None:
            raise Exception("logit bias and log probs not supported")
