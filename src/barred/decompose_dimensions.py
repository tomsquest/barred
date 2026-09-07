import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from barred.exceptions import DecompositionError
from barred.llm import LLM
from barred.types import (
    Criterion,
    DecomposedDimension,
    Dimension,
    Example,
    Instantiation,
    Message,
)


async def decompose_dimensions(
    llm: LLM,
    *,
    criterion: Criterion,
    examples: list[Example],
    concurrency: int = 5,
) -> list[DecomposedDimension]:
    """Decompose `criterion` into deduplicated dimensions with their instantiations.

    Step 1 of the pipeline, in three passes: candidate dimensions are proposed from the
    criterion and its examples, redundant ones are merged in a second pass that replays
    the same conversation, then each surviving dimension is expanded into concrete
    instantiations, in parallel.

    Instantiations are produced with Verbalized Sampling: the model must assign each one
    a polarity and a probability score. Having to place them in a distribution is what
    pushes it past its few default answers; only the descriptions are kept.

    Args:
        llm: Model used for every call.
        criterion: The property samples are labeled against.
        examples: Unlabeled input blocks showing the domain, structure and style to
            stay within.
        concurrency: Dimensions expanded into instantiations in parallel.

    Returns:
        One entry per deduplicated dimension, each with at least one instantiation.

    Raises:
        ValueError: An argument is empty or out of range.
        DecompositionError: The model returned an empty list where entries were needed.
        LLMCallError: An LLM call exhausted its retries.
    """

    if not criterion:
        msg = "criterion must not be empty"
        raise ValueError(msg)
    if not examples:
        msg = "examples must not be empty"
        raise ValueError(msg)
    if concurrency < 1:
        msg = "concurrency must be >= 1"
        raise ValueError(msg)

    dimension_messages = await _generate_dimensions(llm, criterion, examples)
    dimensions, dimension_messages = await _deduplicate_dimensions(
        llm, dimension_messages
    )
    decomposed_dimensions = await _decompose_dimensions(
        llm, dimension_messages, dimensions, concurrency
    )

    llm.observer.on_dimensions_decomposed(
        criterion=criterion,
        examples=examples,
        dimensions=decomposed_dimensions,
    )
    return decomposed_dimensions


class InstantiationRaw(BaseModel):
    """A concrete, tangible case along a dimension, from which test cases for the CRITERION can be constructed."""

    description: str = Field(
        ...,
        description=(
            "A concrete case the dimension can take, stated as a property of the input block. "
            "It must be understandable on its own, by a reader who has no access to the dimension it belongs to. "
            "When the case implies a specific reading of the CRITERION, state that reading explicitly."
        ),
    )
    polarity: Literal["true", "false", "both"] = Field(
        ...,
        description=(
            "Which value of the CRITERION this instantiation can produce on an input block that exhibits it: "
            "'true' if such an input block necessarily satisfies the CRITERION, "
            "'false' if it necessarily fails it, "
            "'both' if either outcome is possible depending on how the case is realized. "
            "Judge the case on its own, not the dimension it belongs to."
        ),
    )
    score: float = Field(
        ...,
        description=(
            "Probability of this instantiation being drawn from the distribution of instantiations for the given dimension"
        ),
        ge=0.0,
        le=1.0,
    )


class DimensionList(BaseModel):
    dimensions: list[Dimension]


class InstantiationList(BaseModel):
    instantiations: list[InstantiationRaw]


def _get_dimension_system_message() -> str:
    return """
You are part of a test case generation system, whose goal is to create test cases for testing a classifier for the following CRITERION.
Each test case is composed of an input block and a label.
Your expertise is creating the key dimensions upon which all test cases will be generated.

Guidelines:
- Extract only dimensions that affect the value of the given CRITERION.
- Think about diverse dimensions that an evaluator would like to test with respect to the given CRITERION.
- Ensure that all dimensions are self-contained, meaning that a reader could understand
why the dimension is relevant without needing outside assumptions.
- Do not suggest dimensions that imply different structure or metadata of the EXAMPLE_INPUT_BLOCK.
- In the description of the dimensions, do not mention the evaluator
"""


def _get_dimension_user_message(criterion: Criterion, examples: list[Example]) -> str:
    return f"""\
<CRITERION>
{criterion}
</CRITERION>

<EXAMPLE_INPUT_BLOCK>
{"\n\n".join(s for s in examples)}
</EXAMPLE_INPUT_BLOCK>
"""


def _get_dimension_dedup_system_message() -> str:
    return """\
You previously proposed a list of candidate dimensions for generating test cases.
Some of these dimensions may be semantically redundant: they describe the same
underlying axis of variation using different wording, or one is a special case
fully subsumed by another.

Your task: produce a deduplicated list of dimensions.
- Merge dimensions that test the same underlying axis of variation into a single dimension.
- When merging, keep the clearer name/description, or write a new one that covers both.
- Keep dimensions distinct if they affect the CRITERION independently, even if related.
- Do not invent new dimensions; only merge or remove from the list you already proposed.
"""


def _get_dimension_dedup_user_message() -> str:
    return "Please return the deduplicated list of dimensions."


def _get_instantiation_system_message() -> str:
    return """\
For the following dimension, please create all reasonable instantiations
that could be used to construct test cases for the CRITERION.

For every instantiation, return whether it is relevant to a True value of the CRITERION, a False value, or Both,
and a score between 0 and 1 reflecting the probability of being drawn from the distribution of instantiations
for the given dimension.
"""


def _get_instantiation_user_message(dimension: Dimension) -> str:
    return f"Dimension: {dimension.name}"


async def _generate_dimensions(
    llm: LLM,
    criterion: Criterion,
    examples: list[Example],
) -> list[Message]:
    """Generate a list of candidate dimensions. Returns the conversation messages for followup."""

    input_messages: list[Message] = [
        {
            "role": "system",
            "content": _get_dimension_system_message(),
        },
        {
            "role": "user",
            "content": _get_dimension_user_message(criterion, examples),
        },
    ]

    result = await llm.call(
        temperature=1.0,
        seed=0,
        response_format=DimensionList,
        messages=input_messages,
        context="decompose_dimensions:generate_dimensions",
    )

    # Assert ok
    dimension_list: DimensionList = result.parsed
    if len(dimension_list.dimensions) == 0:
        msg = "No dimensions generated"
        raise DecompositionError(msg)

    llm.observer.on_dimensions_generated(
        criterion=criterion,
        examples=examples,
        dimensions=dimension_list.dimensions,
    )

    return [*input_messages, result.message]


async def _deduplicate_dimensions(
    llm: LLM,
    previous_messages: list[Message],
) -> tuple[list[Dimension], list[Message]]:
    """Deduplicate dimensions using previous messages. Returns the deduplicated dimensions and messages for followup."""

    messages: list[Message] = [
        *previous_messages,
        {"role": "system", "content": _get_dimension_dedup_system_message()},
        {"role": "user", "content": _get_dimension_dedup_user_message()},
    ]

    result = await llm.call(
        temperature=0.0,  # Determinist
        seed=0,
        response_format=DimensionList,
        messages=messages,
        context="decompose_dimensions:deduplicate_dimensions",
    )

    dimension_list: DimensionList = result.parsed
    if len(dimension_list.dimensions) == 0:
        msg = "Unable to deduplicate dimensions"
        raise DecompositionError(msg)

    llm.observer.on_dimensions_deduplicated(dimensions=dimension_list.dimensions)

    return dimension_list.dimensions, [*messages, result.message]


async def _decompose_dimensions(
    llm: LLM,
    previous_messages: list[Message],
    dimensions: list[Dimension],
    concurrency: int,
) -> list[DecomposedDimension]:
    semaphore = asyncio.Semaphore(concurrency)

    async def decompose_one(dimension: Dimension) -> DecomposedDimension:
        async with semaphore:
            instantiations = await _generate_instantiations(
                llm, previous_messages, dimension
            )

        return DecomposedDimension(
            dimension=dimension,
            instantiations=[
                Instantiation(description=i.description)
                for i in instantiations.instantiations
            ],
        )

    try:
        async with asyncio.TaskGroup() as task_group:
            tasks = [task_group.create_task(decompose_one(d)) for d in dimensions]
    except BaseExceptionGroup as group:
        # Re-raise the first one so callers see DecompositionError and LLMCallError directly (not a group)
        raise group.exceptions[0] from group

    return [task.result() for task in tasks]


async def _generate_instantiations(
    llm: LLM,
    previous_messages: list[Message],
    dimension: Dimension,
) -> InstantiationList:
    messages: list[Message] = [
        {
            "role": "system",
            "content": _get_instantiation_system_message(),
        },
        {
            "role": "user",
            "content": _get_instantiation_user_message(dimension),
        },
    ]

    instantiation_response = await llm.call(
        temperature=1.0,
        seed=0,
        response_format=InstantiationList,
        messages=[
            *previous_messages,
            *messages,
        ],
        context=f"decompose_dimensions:generate_instantiations:dim={dimension.name}",
    )

    instantiation_list: InstantiationList = instantiation_response.parsed
    if len(instantiation_list.instantiations) == 0:
        msg = "No instantiations generated"
        raise DecompositionError(msg)

    llm.observer.on_instantiations_generated(
        dimension=dimension,
        instantiations=[
            Instantiation(description=i.description)
            for i in instantiation_list.instantiations
        ],
    )

    return instantiation_list
