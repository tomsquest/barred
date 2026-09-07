from types import SimpleNamespace
from typing import Any, cast

from barred.generate_sample import generate_sample
from barred.llm import LLM
from barred.observer import NullObserver
from barred.types import Instantiation, Sample

INSTANTIATION = Instantiation(description="a compliment meant sarcastically")
SAMPLE = Sample(reasoning="reasoning", input_block="input block", label=True)


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.sample: Sample | None = None
        self.target_verdict: bool | None = None

    def on_sample_generated(
        self, *, sample: Sample, target_verdict: bool, **_: object
    ) -> None:
        self.sample = sample
        self.target_verdict = target_verdict


class _FakeLLM:
    def __init__(self) -> None:
        self.observer = _RecordingObserver()
        self.kwargs: dict[str, Any] = {}

    async def call(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(parsed=SAMPLE)


async def test_generate_sample() -> None:
    llm = _FakeLLM()

    sample = await generate_sample(
        cast("LLM", llm),
        criterion="criterion",
        example="example",
        instantiation=INSTANTIATION,
        target_verdict=True,
    )

    assert sample == SAMPLE
    assert llm.observer.sample == SAMPLE
    assert llm.observer.target_verdict is True


async def test_no_seed() -> None:
    llm = _FakeLLM()

    await generate_sample(
        cast("LLM", llm),
        criterion="criterion",
        example="example",
        instantiation=INSTANTIATION,
        target_verdict=True,
    )

    # a fixed seed would collapse recurring prompts into duplicates
    assert llm.kwargs["seed"] is None
