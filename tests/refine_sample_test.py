from types import SimpleNamespace
from typing import Any, cast

from barred.llm import LLM
from barred.observer import NullObserver
from barred.refine_sample import refine_sample
from barred.types import Instantiation, Sample

INSTANTIATION = Instantiation(description="a compliment meant sarcastically")
REJECTED = Sample(reasoning="reasoning", input_block="input block", label=True)
FEEDBACK = ["the input block reads as a plain compliment"]


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.previous: Sample | None = None
        self.refined: Sample | None = None
        self.dissenting_feedback: list[str] = []

    def on_sample_refined(
        self,
        *,
        previous: Sample,
        refined: Sample,
        dissenting_feedback: list[str],
        **_: object,
    ) -> None:
        self.previous = previous
        self.refined = refined
        self.dissenting_feedback = dissenting_feedback


class _FakeLLM:
    def __init__(self, refined: Sample) -> None:
        self.observer = _RecordingObserver()
        self.refined = refined
        self.kwargs: dict[str, Any] = {}

    async def call(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(parsed=self.refined)


async def test_refine_sample() -> None:
    llm = _FakeLLM(
        Sample(reasoning="new reasoning", input_block="new input block", label=True)
    )

    refined = await refine_sample(
        cast("LLM", llm),
        criterion="criterion",
        example="example",
        instantiation=INSTANTIATION,
        sample=REJECTED,
        dissenting_feedback=FEEDBACK,
    )

    assert refined.input_block == "new input block"
    assert llm.observer.previous == REJECTED
    assert llm.observer.refined == refined
    assert llm.observer.dissenting_feedback == FEEDBACK


async def test_label_is_preserved() -> None:
    llm = _FakeLLM(
        # the model flipped the label
        Sample(reasoning="new reasoning", input_block="new input block", label=False)
    )

    refined = await refine_sample(
        cast("LLM", llm),
        criterion="criterion",
        example="example",
        instantiation=INSTANTIATION,
        sample=REJECTED,
        dissenting_feedback=FEEDBACK,
    )

    assert refined.label is True


async def test_no_seed() -> None:
    llm = _FakeLLM(
        Sample(reasoning="new reasoning", input_block="new input block", label=True)
    )

    await refine_sample(
        cast("LLM", llm),
        criterion="criterion",
        example="example",
        instantiation=INSTANTIATION,
        sample=REJECTED,
        dissenting_feedback=FEEDBACK,
    )

    # a fixed seed would collapse recurring prompts into duplicates
    assert llm.kwargs["seed"] is None
