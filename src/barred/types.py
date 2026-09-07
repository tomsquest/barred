from typing import Any, Literal

from any_llm.types.completion import ChatCompletionMessage
from pydantic import BaseModel, Field

Message = dict[str, Any] | ChatCompletionMessage
"""One turn of a conversation sent to the model.

Either a plain `{"role": ..., "content": ...}` mapping, or an assistant message returned
by a previous call, appended as-is to continue that conversation.
"""

Criterion = str
"""The property every sample is labeled against, stated so it can be answered True or False.

It drives the whole run: dimensions are extracted from it, and judges rule on it.
"""

Example = str
"""An unlabeled input block, showing the domain, structure and style to stay within.

Only used as a stylistic anchor: its own label is never needed nor inferred.
"""


class Dimension(BaseModel):
    """An axis of variation of the domain along which test cases for the CRITERION will be generated."""

    name: str = Field(
        ...,
        description="Short handle, 2-5 words",
    )
    description: str = Field(
        ...,
        description="Self-contained explanation of why this axis affects the value of the CRITERION",
    )


class Instantiation(BaseModel):
    """A concrete case along a dimension, from which test cases for the CRITERION can be constructed."""

    description: str


class DecomposedDimension(BaseModel):
    """A dimension together with the concrete instantiations it was expanded into."""

    dimension: Dimension
    instantiations: list[Instantiation]


class Sample(BaseModel):
    """A generated test case for the CRITERION, composed of an input block and the label it should get."""

    reasoning: str = Field(
        ...,
        description="Why the input block should get this label for the CRITERION",
    )
    input_block: str = Field(
        ...,
        description="The generated input block, matching the domain and style of the EXAMPLE_INPUT_BLOCK",
    )
    label: bool = Field(
        ...,
        description="True if the CRITERION holds on the generated input block, False otherwise",
    )


class JudgeVerdict(BaseModel):
    """A judge's assessment of whether the CRITERION holds on the input block."""

    reasoning: str = Field(
        ...,
        description="Why the CRITERION holds or doesn't hold on the input block",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Confidence level in the classification",
    )
    label: bool = Field(
        ...,
        description="True if the CRITERION holds on the input block, False otherwise",
    )


class DebateResult(BaseModel):
    """The outcome of a debate over a sample: whether it is valid, with dissenting feedback."""

    valid: bool
    dissenting_feedback: list[str]


class RefinementRound(BaseModel):
    """One validation round: the sample as it stood, and the debate outcome on it."""

    sample: Sample
    valid: bool
    dissenting_feedback: list[str]


class GenerationRecord(BaseModel):
    """Full trace of one attempt, accepted or rejected."""

    attempt: int
    dimension: str
    instantiation: str
    example: Example
    target_verdict: bool
    accepted: bool
    refinement_rounds: list[RefinementRound]
    final_sample: Sample
