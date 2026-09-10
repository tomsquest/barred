from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import (
    ContentFilterFinishReasonError,
    GatewayTimeoutError,
    LengthFinishReasonError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ParsedChatCompletion,
    ReasoningEffort,
)
from pydantic import BaseModel, ValidationError

from barred.exceptions import LLMCallError
from barred.observer import NullObserver, Observer
from barred.types import Message

# any-llm only raises its unified exceptions (RateLimitError, ProviderError...) when this is
# set in the environment; otherwise it lets the raw provider exception through and warns.
# The retry policy below is written against those unified exceptions, so the library forces
# the setting rather than leaving its retries silently dead in consumer environments.
# Issue: https://github.com/mozilla-ai/any-llm/issues/1369
os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"


class TokenUsage(BaseModel):
    """Token accounting for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse[T: BaseModel](BaseModel):
    """Result of one `LLM.call`."""

    parsed: T
    """The structured output, validated against the requested `response_format`."""

    message: ChatCompletionMessage
    """The raw assistant message, so it can be appended to `messages` for a followup turn."""

    usage: TokenUsage
    """Usage in terms of tokens"""


class LLM:
    """Configure a model once, then call it for structured output.

    Every step of the pipeline shares a single instance, so the model, its reasoning
    effort and its retry policy are decided in one place. Token usage accumulates here
    across every call, and this is also what carries the `observer` the whole pipeline
    reports to.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        provider: str | LLMProvider,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 16384,  # Give thinking + the visible output real headroom
        reasoning_effort: ReasoningEffort = "medium",  # Thinking level used in BARRED
        retry: int = 2,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        observer: Observer | None = None,
        **provider_kwargs: Any | None,  # noqa: ANN401
    ) -> None:
        """
        Args:
            provider: Which any-llm provider to route calls through.
            model: Model id, used by every `call`.
            api_key: API key for the provider.
            api_base: API base URL for the provider.
            max_tokens: Passed to every `call`.
            reasoning_effort: Passed to every `call`.
            retry: Default number of retries on transient failure (0 = no retry).
            retry_base_delay: Base delay in seconds for exponential backoff between retries.
            retry_max_delay: Ceiling in seconds for a single backoff wait, so the last
                retries stay short instead of doubling into minutes.
            observer: Notified of every pipeline event, not just LLM calls. Defaults to a
                no-op observer; `LoggingObserver` logs the whole run.
            **provider_kwargs: Forwarded to `AnyLLM.create` (e.g., VertexAI `project`, `location`).

        Raises:
            ValueError: An argument is empty or out of range.
        """
        if not model:
            msg = "model must not be empty"
            raise ValueError(msg)
        if max_tokens < 1:
            msg = "max_tokens must be >= 1"
            raise ValueError(msg)
        if retry < 0:
            msg = "retry must be >= 0"
            raise ValueError(msg)
        if retry_base_delay <= 0:
            msg = "retry_base_delay must be > 0"
            raise ValueError(msg)
        if retry_max_delay < retry_base_delay:
            msg = "retry_max_delay must be >= retry_base_delay"
            raise ValueError(msg)

        self._client = AnyLLM.create(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            **provider_kwargs,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._retry = retry
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self.observer = observer or NullObserver()
        self.total_usage = TokenUsage()

    async def call[T: BaseModel](
        self,
        *,
        messages: list[Message],
        response_format: type[T],
        temperature: float | None = None,
        seed: int | None = None,
        context: str = "",
    ) -> LLMResponse[T]:
        """Call the model and parse its output as `response_format`.

        Retries with exponential backoff on transient provider errors (rate limits,
        upstream/gateway failures...), when the provider returns no parsable content, and
        when the output does not match the requested schema.

        Args:
            messages: Conversation sent to the model. Assistant messages returned by a
                previous call can be appended to continue that conversation.
            response_format: Schema the output is validated against, and the type of
                `LLMResponse.parsed`. Its docstrings and field descriptions are part of
                the prompt the model receives.
            temperature: Sampling temperature. `None` leaves the provider default.
            seed: Fixed seed for reproducible output when the provider supports it.
                `None` lets identical prompts produce different outputs, which the
                generation steps rely on.
            context: Free-form label describing where this call originates (e.g.
                "debate:round1:judge=strict"). Forwarded verbatim to the observer's
                `on_llm_call` so raw prompts can be tied back to their business context.

        Raises:
            ValueError: An argument is empty or out of range.
            LLMCallError: Every attempt failed on a retryable error.
        """
        if not messages:
            msg = "messages must not be empty"
            raise ValueError(msg)
        if temperature is not None and temperature < 0:
            msg = "temperature must be >= 0"
            raise ValueError(msg)

        last_error: Exception
        for attempt in range(self._retry + 1):
            try:
                completion = await self._client.acompletion(
                    model=self._model,
                    temperature=temperature,
                    max_tokens=self._max_tokens,
                    seed=seed,
                    reasoning_effort=self._reasoning_effort,
                    response_format=response_format,
                    messages=messages,
                )
                response = self._handle_completion(completion, response_format)
                self.observer.on_llm_call(
                    context=context,
                    messages=messages,
                    response=response.parsed,
                )
                return response
            except _RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt == self._retry:
                    break
                backoff = min(
                    self._retry_base_delay * (2**attempt), self._retry_max_delay
                )
                delay = backoff + random.uniform(0, 0.5)  # noqa: S311
                self.observer.on_llm_retry(
                    context=context,
                    attempt=attempt + 1,
                    max_attempts=self._retry + 1,
                    error=e,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        msg = f"LLM call to {self._model!r} failed after {self._retry + 1} attempt(s)"
        raise LLMCallError(msg) from last_error

    def _handle_completion[T: BaseModel](
        self,
        completion: ParsedChatCompletion[T],
        response_format: type[T],
    ) -> LLMResponse[T]:
        message = completion.choices[0].message
        parsed: T | None = message.parsed
        if parsed is None:
            msg = f"Provider returned no parsable {response_format.__name__}"
            raise _NoParsableContentError(msg)

        usage = self._extract_token_usage(completion)
        self.total_usage += usage

        return LLMResponse(parsed=parsed, message=message, usage=usage)

    def _extract_token_usage(self, completion: ChatCompletion) -> TokenUsage:
        usage = completion.usage
        if usage is None:
            return TokenUsage()

        cached_tokens = 0
        if usage.prompt_tokens_details is not None:
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

        return TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=usage.total_tokens,
        )


class _NoParsableContentError(Exception):
    """The provider returned a response with no content to parse."""


# Transient failures worth retrying
_RETRYABLE_EXCEPTIONS = (
    # AnyLLM errors
    RateLimitError,
    ProviderError,
    UpstreamProviderError,
    GatewayTimeoutError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    # The provider returned no parsable output (empty output).
    _NoParsableContentError,
    # The model produced output that does not match the requested schema.
    ValidationError,
)
