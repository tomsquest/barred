from types import SimpleNamespace
from typing import cast

import pytest

from barred.decompose_dimensions import (
    DimensionList,
    InstantiationList,
    InstantiationRaw,
    decompose_dimensions,
)
from barred.exceptions import DecompositionError
from barred.llm import LLM
from barred.observer import NullObserver
from barred.types import DecomposedDimension, Dimension, Instantiation

CRITERION = "criterion"
EXAMPLES = ["example"]


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.generated: list[str] = []
        self.deduplicated: list[str] = []
        self.instantiated: list[str] = []
        self.decomposed: list[str] = []

    def on_dimensions_generated(
        self, *, dimensions: list[Dimension], **_: object
    ) -> None:
        self.generated = [dimension.name for dimension in dimensions]

    def on_dimensions_deduplicated(
        self, *, dimensions: list[Dimension], **_: object
    ) -> None:
        self.deduplicated = [dimension.name for dimension in dimensions]

    def on_instantiations_generated(
        self,
        *,
        dimension: Dimension,
        instantiations: list[Instantiation],  # noqa: ARG002
        **_: object,
    ) -> None:
        self.instantiated.append(dimension.name)

    def on_dimensions_decomposed(
        self, *, dimensions: list[DecomposedDimension], **_: object
    ) -> None:
        self.decomposed = [entry.dimension.name for entry in dimensions]


class _FakeLLM:
    def __init__(
        self,
        *,
        candidates: list[str],
        deduplicated: list[str],
        instantiations: list[str],
    ) -> None:
        self.observer = _RecordingObserver()
        self.dimension_lists = [candidates, deduplicated]
        self.instantiations = instantiations

    async def call(self, *, response_format: type, **_: object) -> SimpleNamespace:
        if response_format is DimensionList:
            parsed = DimensionList(
                dimensions=[
                    Dimension(name=name, description="description")
                    for name in self.dimension_lists.pop(0)
                ]
            )
        else:
            parsed = InstantiationList(
                instantiations=[
                    InstantiationRaw(
                        description=description, polarity="both", score=0.5
                    )
                    for description in self.instantiations
                ]
            )

        return SimpleNamespace(
            parsed=parsed, message={"role": "assistant", "content": ""}
        )


async def test_decompose() -> None:
    llm = _FakeLLM(
        # the first two candidates say the same thing
        candidates=["tone", "tone again", "length"],
        deduplicated=["tone", "length"],
        instantiations=["short", "long"],
    )

    dimensions = await decompose_dimensions(
        cast("LLM", llm), criterion=CRITERION, examples=EXAMPLES
    )

    assert [entry.dimension.name for entry in dimensions] == ["tone", "length"]
    assert [i.description for i in dimensions[0].instantiations] == ["short", "long"]
    assert llm.observer.generated == ["tone", "tone again", "length"]
    assert llm.observer.deduplicated == ["tone", "length"]
    assert llm.observer.instantiated == ["tone", "length"]
    assert llm.observer.decomposed == ["tone", "length"]


async def test_no_dimension_generated() -> None:
    llm = _FakeLLM(candidates=[], deduplicated=[], instantiations=[])

    with pytest.raises(DecompositionError, match="No dimensions generated"):
        await decompose_dimensions(
            cast("LLM", llm), criterion=CRITERION, examples=EXAMPLES
        )


async def test_no_dimension_left_after_deduplication() -> None:
    llm = _FakeLLM(candidates=["tone"], deduplicated=[], instantiations=[])

    with pytest.raises(DecompositionError, match="Unable to deduplicate dimensions"):
        await decompose_dimensions(
            cast("LLM", llm), criterion=CRITERION, examples=EXAMPLES
        )


async def test_no_instantiation_generated() -> None:
    llm = _FakeLLM(candidates=["tone"], deduplicated=["tone"], instantiations=[])

    with pytest.raises(DecompositionError, match="No instantiations generated"):
        await decompose_dimensions(
            cast("LLM", llm), criterion=CRITERION, examples=EXAMPLES
        )
