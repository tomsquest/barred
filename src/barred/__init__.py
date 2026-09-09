"""Generate faithful and diverse labeled samples from a criterion and a few unlabeled examples."""

from barred.debate import debate
from barred.decompose_dimensions import decompose_dimensions
from barred.exceptions import BarredError, DecompositionError, LLMCallError
from barred.generate_sample import generate_sample
from barred.llm import LLM, LLMResponse, TokenUsage
from barred.observer import LoggingObserver, NullObserver, Observer
from barred.pipeline import barred
from barred.refine_sample import refine_sample
from barred.types import (
    Criterion,
    DebateResult,
    DecomposedDimension,
    Dimension,
    Example,
    GenerationRecord,
    Instantiation,
    JudgeVerdict,
    Message,
    RefinementRound,
    Sample,
)

__all__ = [
    "LLM",
    "BarredError",
    "Criterion",
    "DebateResult",
    "DecomposedDimension",
    "DecompositionError",
    "Dimension",
    "Example",
    "GenerationRecord",
    "Instantiation",
    "JudgeVerdict",
    "LLMCallError",
    "LLMResponse",
    "LoggingObserver",
    "Message",
    "NullObserver",
    "Observer",
    "RefinementRound",
    "Sample",
    "TokenUsage",
    "barred",
    "debate",
    "decompose_dimensions",
    "generate_sample",
    "refine_sample",
]
