from typing import Any

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import AuthenticationError, RateLimitError
from any_llm.types.completion import (
    ChatCompletion,
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
    ParsedChoice,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from pydantic import BaseModel, ValidationError

from barred.exceptions import LLMCallError
from barred.llm import LLM, _NoParsableContentError
from barred.observer import NullObserver, Observer


class _Answer(BaseModel):
    value: int


ANSWER = _Answer(value=42)


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff adds a random jitter: drop it so delays are exact and nothing really sleeps."""
    monkeypatch.setattr("barred.llm.random.uniform", lambda _a, _b: 0.0)


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.calls: list[tuple[str, BaseModel]] = []
        self.retries: list[tuple[int, int, type[Exception]]] = []
        self.delays: list[float] = []

    def on_llm_call(self, *, context: str, response: BaseModel, **_: object) -> None:
        self.calls.append((context, response))

    def on_llm_retry(
        self,
        *,
        attempt: int,
        max_attempts: int,
        error: Exception,
        delay: float,
        **_: object,
    ) -> None:
        self.retries.append((attempt, max_attempts, type(error)))
        self.delays.append(delay)


class _FakeClient:
    """Stands in for `AnyLLM`: hands out canned completions, or raises canned errors.

    Records the kwargs of every attempt, so `len(client.attempts)` is the call count.
    """

    def __init__(self, responses: list[ChatCompletion | Exception]) -> None:
        self._responses = list(responses)
        self.attempts: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> ChatCompletion:
        self.attempts.append(kwargs)
        if not self._responses:
            msg = f"the fake ran out of responses after {len(self.attempts)} call(s)"
            raise AssertionError(msg)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _completion(
    parsed: _Answer | None,
    *,
    usage: CompletionUsage | None = None,
) -> ChatCompletion:
    message = ParsedChatCompletionMessage[_Answer](
        role="assistant",
        content=parsed.model_dump_json() if parsed is not None else None,
        parsed=parsed,
    )
    return ParsedChatCompletion[_Answer](
        id="1",
        choices=[ParsedChoice[_Answer](finish_reason="stop", index=0, message=message)],
        created=0,
        model="some-model-name",
        object="chat.completion",
        usage=usage,
    )


def _validation_error() -> ValidationError:
    """A real ValidationError, as pydantic raises when the model breaks the schema."""
    try:
        _Answer.model_validate({"value": "not-an-int"})
    except ValidationError as error:
        return error
    msg = "expected the payload to fail validation"
    raise AssertionError(msg)


def _llm(
    responses: list[ChatCompletion | Exception],
    *,
    retry: int | None = None,
    observer: Observer | None = None,
) -> tuple[LLM, _FakeClient]:
    llm = LLM(
        provider=LLMProvider.GEMINI,
        model="some-model-name",
        api_key="fake",  # pragma: allowlist secret
        retry=0 if retry is None else retry,
        retry_base_delay=0.001,
        observer=observer,
    )
    client = _FakeClient(responses)
    llm._client = client  # noqa: SLF001  # ty:ignore[invalid-assignment]
    return llm, client


class TestRequestAndResponse:
    async def test_forwards_the_call_settings(self) -> None:
        llm, client = _llm([_completion(ANSWER)])

        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            temperature=0.3,
            seed=7,
        )

        sent = client.attempts[0]
        assert sent["model"] == "some-model-name"
        assert sent["response_format"] is _Answer
        assert sent["temperature"] == 0.3
        assert sent["seed"] == 7
        assert sent["max_tokens"] == 16384
        assert sent["reasoning_effort"] == "medium"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]

    async def test_returns_the_parsed_output_and_the_raw_message(self) -> None:
        llm, client = _llm([_completion(ANSWER)])

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.parsed == ANSWER
        assert result.message.content == '{"value":42}'
        assert len(client.attempts) == 1

    async def test_reports_a_successful_call_to_the_observer(self) -> None:
        observer = _RecordingObserver()
        llm, _ = _llm([_completion(ANSWER)], observer=observer)

        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            context="debate:round1",
        )

        assert observer.calls == [("debate:round1", ANSWER)]
        assert observer.retries == []


class TestTokenUsage:
    async def test_extracts_every_field(self) -> None:
        llm, _ = _llm(
            [
                _completion(
                    ANSWER,
                    usage=CompletionUsage(
                        prompt_tokens=1,
                        completion_tokens=2,
                        total_tokens=3,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
                    ),
                )
            ]
        )

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2
        assert result.usage.cached_tokens == 4
        assert result.usage.total_tokens == 3
        assert llm.total_usage == result.usage

    async def test_accumulates_across_calls(self) -> None:
        llm, _ = _llm(
            [
                _completion(
                    ANSWER,
                    usage=CompletionUsage(
                        prompt_tokens=1,
                        completion_tokens=2,
                        total_tokens=3,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
                    ),
                ),
                _completion(
                    ANSWER,
                    usage=CompletionUsage(
                        prompt_tokens=1,
                        completion_tokens=2,
                        total_tokens=3,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
                    ),
                ),
            ]
        )

        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert llm.total_usage.input_tokens == 2
        assert llm.total_usage.output_tokens == 4
        assert llm.total_usage.total_tokens == 6
        assert llm.total_usage.cached_tokens == 8

    async def test_is_zero_when_the_provider_reports_none(self) -> None:
        llm, _ = _llm([_completion(ANSWER, usage=None)])

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.usage.total_tokens == 0
        assert llm.total_usage.total_tokens == 0

    async def test_cached_tokens_without_prompt_details(self) -> None:
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        llm, _ = _llm([_completion(ANSWER, usage=usage)])

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.usage.cached_tokens == 0


class TestRetryPolicy:
    async def test_transient_error_then_success(self) -> None:
        observer = _RecordingObserver()
        llm, client = _llm(
            [RateLimitError(), RateLimitError(), _completion(ANSWER)],
            observer=observer,
            retry=2,
        )

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            context="debate:round1",
        )

        assert result.parsed == ANSWER
        assert len(client.attempts) == 3
        assert observer.retries == [
            (1, 3, RateLimitError),
            (2, 3, RateLimitError),
        ]
        # exponential backoff on a 0.001s base, jitter dropped by the fixture
        assert observer.delays == [0.001, 0.002]
        # the failed attempts are not reported as calls
        assert observer.calls == [("debate:round1", ANSWER)]

    async def test_no_parsable_content_then_success(self) -> None:
        llm, client = _llm(
            [_completion(None), _completion(ANSWER)],
            retry=2,
        )

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.parsed == ANSWER
        assert len(client.attempts) == 2

    async def test_broken_schema_then_success(self) -> None:
        """Malformed structured output is worth another call: it often validates next time."""
        llm, client = _llm(
            [_validation_error(), _completion(ANSWER)],
            retry=2,
        )

        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
        )

        assert result.parsed == ANSWER
        assert len(client.attempts) == 2

    async def test_authentication_error_is_not_retried(self) -> None:
        observer = _RecordingObserver()
        llm, client = _llm(
            [AuthenticationError("bad key"), _completion(ANSWER)],
            observer=observer,
        )

        with pytest.raises(AuthenticationError):
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
            )

        assert len(client.attempts) == 1
        assert observer.retries == []

    async def test_unrelated_value_error_is_not_retried(self) -> None:
        """ValueError used to be retryable wholesale; only the precise cases are now."""
        llm, client = _llm([ValueError("some unrelated bug"), _completion(ANSWER)])

        with pytest.raises(ValueError, match="some unrelated bug"):
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
            )

        assert len(client.attempts) == 1

    async def test_retry_disabled(self) -> None:
        observer = _RecordingObserver()
        llm, client = _llm([RateLimitError("boom")], retry=0, observer=observer)

        with pytest.raises(LLMCallError, match="failed after 1 attempt"):
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
            )

        assert len(client.attempts) == 1
        assert observer.retries == []

    async def test_every_attempt_fails(self) -> None:
        retry = 4
        attempts = retry + 1
        last = RateLimitError("boom")
        llm, client = _llm(
            [
                RateLimitError("boom"),
                RateLimitError("boom"),
                RateLimitError("boom"),
                RateLimitError("boom"),
                last,
            ],
            retry=retry,
        )

        with pytest.raises(
            LLMCallError, match=f"failed after {attempts} attempt"
        ) as excinfo:
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
            )

        assert len(client.attempts) == attempts
        assert excinfo.value.__cause__ is last

    async def test_unparsable_content(self) -> None:
        llm, _ = _llm([_completion(None)])

        with pytest.raises(LLMCallError) as excinfo:
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
            )

        assert isinstance(excinfo.value.__cause__, _NoParsableContentError)
