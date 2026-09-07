class BarredError(Exception):
    """Base class for every error raised by this library."""


class LLMCallError(BarredError):
    """An LLM call kept failing on retryable errors until its retries ran out.

    The last failure is available as `__cause__`.
    """


class DecompositionError(BarredError):
    """The model returned an empty list where `decompose_dimensions` needed entries."""
