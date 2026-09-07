import asyncio
import itertools
from collections.abc import AsyncGenerator, Awaitable, Callable
from types import SimpleNamespace
from typing import cast

import pytest
from any_llm.exceptions import AuthenticationError

from barred.llm import LLM
from barred.observer import NullObserver, Observer
from barred.pipeline import barred
from barred.types import (
    DebateResult,
    DecomposedDimension,
    Dimension,
    GenerationRecord,
    Instantiation,
    Sample,
)

DECOMPOSED_DIMENSION = DecomposedDimension(
    dimension=Dimension(name="dim", description="d"),
    instantiations=[Instantiation(description="inst")],
)
SAMPLE = Sample(reasoning="r", input_block="b", label=True)
REFINED = Sample(reasoning="refined r", input_block="refined b", label=True)


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.records: list[GenerationRecord] = []

    def on_sample_finalized(self, *, record: GenerationRecord, **_: object) -> None:
        self.records.append(record)


def _debate(verdicts: list[bool]) -> Callable[..., Awaitable[DebateResult]]:
    """Return each verdict in turn, then keep returning the last one.

    A multi-verdict sequence describes a single attempt, so use `num_samples=1` to keep
    concurrent attempts from consuming it, or `concurrency=1` to spread it over
    successive attempts.
    """
    remaining = list(verdicts)

    async def _run(**_: object) -> DebateResult:
        valid = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return DebateResult(
            valid=valid,
            dissenting_feedback=[] if valid else ["some dissenting feedback"],
        )

    return _run


def _debate_discarding_one_in(n: int) -> Callable[..., Awaitable[DebateResult]]:
    """Discard one debate in `n`, whatever the order the calls land in.

    Unlike `_debate`, no attempt depends on getting a particular call, so the run
    stays deterministic in aggregate when attempts interleave.
    """
    calls = itertools.count()

    async def _run(**_: object) -> DebateResult:
        valid = next(calls) % n != 0
        return DebateResult(
            valid=valid,
            dissenting_feedback=[] if valid else ["some dissenting feedback"],
        )

    return _run


async def _generate_sample(**_: object) -> Sample:
    return SAMPLE


async def _refine_sample_noop(*, sample: Sample, **_: object) -> Sample:
    return sample


def _barred(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    *,
    debate: Callable[..., Awaitable[DebateResult]],
    generate_sample: Callable[..., Awaitable[Sample]],
    refine_sample: Callable[..., Awaitable[Sample]],
    observer: Observer | None = None,
    num_samples: int,
    dimensions: list[DecomposedDimension] | None = None,
    concurrency: int | None = None,
    max_debate_rounds: int | None = None,
    max_refine_rounds: int | None = None,
    max_attempts: int | None = None,
) -> AsyncGenerator[Sample]:

    # barred() optional params
    optional: dict[str, int] = {}
    if concurrency is not None:
        optional["concurrency"] = concurrency
    if max_debate_rounds is not None:
        optional["max_debate_rounds"] = max_debate_rounds
    if max_refine_rounds is not None:
        optional["max_refine_rounds"] = max_refine_rounds
    if max_attempts is not None:
        optional["max_attempts"] = max_attempts

    if dimensions is None:
        dimensions = [DECOMPOSED_DIMENSION]

    monkeypatch.setattr("barred.pipeline.debate", debate)
    monkeypatch.setattr("barred.pipeline.generate_sample", generate_sample)
    monkeypatch.setattr("barred.pipeline.refine_sample", refine_sample)

    return barred(
        llm=cast("LLM", SimpleNamespace(observer=observer or NullObserver())),
        criterion="crit",
        dimensions=dimensions,
        examples=["example"],
        num_samples=num_samples,
        **optional,
    )


async def test_run_yields_num_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([True]),
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=7,
    )

    samples = [sample async for sample in run]
    assert samples == [SAMPLE] * 7
    assert all(r.accepted for r in observer.records)
    assert sorted(r.attempt for r in observer.records) == [1, 2, 3, 4, 5, 6, 7]


async def test_discarded_attempts_are_not_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([False, True]),
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=1,
        concurrency=1,
        max_refine_rounds=0,
    )

    samples = [sample async for sample in run]
    assert samples == [SAMPLE]
    assert [r.accepted for r in observer.records] == [False, True]


async def test_yields_exactly_num_samples_when_attempts_are_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        # Without the refill cap, the last round would yield 12 samples, not 10.
        debate=_debate_discarding_one_in(4),
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=10,
        concurrency=4,
        max_refine_rounds=0,
    )

    samples = [sample async for sample in run]
    assert samples == [SAMPLE] * 10
    assert len(observer.records) > 10
    assert any(not r.accepted for r in observer.records)


async def test_stops_at_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([False]),  # nothing is ever accepted
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=10,
        concurrency=4,
        # Not a multiple of the concurrency: the last refill owes 2 attempts, not 4
        max_attempts=6,
    )

    samples = [sample async for sample in run]
    assert samples == []
    assert len(observer.records) == 6


async def test_max_attempts_defaults_to_twice_num_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([False]),  # nothing is ever accepted
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=3,
    )

    samples = [sample async for sample in run]
    assert samples == []
    assert len(observer.records) == 6


async def test_yields_what_it_got_when_max_attempts_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate_discarding_one_in(2),
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=10,
        max_refine_rounds=0,
        max_attempts=6,
    )

    # Short of the 10 asked for, and no error: the samples already paid for are kept
    samples = [sample async for sample in run]
    assert samples == [SAMPLE] * 3
    assert len(observer.records) == 6


async def test_raises_given_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _llm_invalid_api_key(**_: object) -> Sample:
        msg = "invalid API key"
        raise AuthenticationError(msg)

    run = _barred(
        monkeypatch,
        debate=_debate([True]),
        generate_sample=_llm_invalid_api_key,
        refine_sample=_refine_sample_noop,
        num_samples=5,
    )

    with pytest.raises(AuthenticationError, match="invalid API key"):
        await anext(run)


async def test_sample_refined_then_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refinements: list[list[str]] = []

    async def _refine(*, dissenting_feedback: list[str], **_: object) -> Sample:
        refinements.append(dissenting_feedback)
        return REFINED

    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([False, True]),
        generate_sample=_generate_sample,
        refine_sample=_refine,
        observer=observer,
        num_samples=1,
    )

    samples = [sample async for sample in run]
    assert samples == [REFINED]
    assert len(observer.records) == 1
    record = observer.records[0]
    assert record.accepted
    assert [r.valid for r in record.refinement_rounds] == [False, True]
    assert record.final_sample == REFINED
    assert refinements == [["some dissenting feedback"]]


async def test_no_refinement_after_the_last_debate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dissenting_feedbacks: list[list[str]] = []

    async def _refine(
        *, sample: Sample, dissenting_feedback: list[str], **_: object
    ) -> Sample:
        dissenting_feedbacks.append(dissenting_feedback)
        return sample

    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        # The first attempt is rejected throughout, the second ends the run
        debate=_debate([False, False, False, True]),
        generate_sample=_generate_sample,
        refine_sample=_refine,
        observer=observer,
        num_samples=1,
        concurrency=1,
    )

    # Don't care about the samples in this test
    async for _sample in run:
        pass

    # max_refine_rounds defaults to 2: 3 debates, and the last one is not refined
    discarded = observer.records[0]
    assert len(discarded.refinement_rounds) == 3
    assert dissenting_feedbacks == [
        ["some dissenting feedback"],
        ["some dissenting feedback"],
    ]


async def test_more_max_refine_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dissenting_feedbacks: list[list[str]] = []

    async def _refine(
        *, sample: Sample, dissenting_feedback: list[str], **_: object
    ) -> Sample:
        dissenting_feedbacks.append(dissenting_feedback)
        return sample

    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([False] * 6 + [True]),
        generate_sample=_generate_sample,
        refine_sample=_refine,
        observer=observer,
        num_samples=1,
        concurrency=1,
        max_refine_rounds=5,
    )

    # Don't care about the samples in this test
    async for _sample in run:
        pass

    # max_refine_rounds + 1 debates, max_refine_rounds refinements
    discarded = observer.records[0]
    assert len(discarded.refinement_rounds) == 6
    assert dissenting_feedbacks == [["some dissenting feedback"]] * 5


async def test_runs_attempts_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = 0
    peak = 0

    async def _generate(**_: object) -> Sample:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)  # let the other starts
        running -= 1
        return SAMPLE

    run = _barred(
        monkeypatch,
        debate=_debate([True]),
        generate_sample=_generate,
        refine_sample=_refine_sample_noop,
        num_samples=10,
        concurrency=8,
    )

    # Don't care about the samples in this test
    async for _sample in run:
        pass

    assert peak == 8


async def test_draws_instantiations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _RecordingObserver()
    run = _barred(
        monkeypatch,
        debate=_debate([True]),
        generate_sample=_generate_sample,
        refine_sample=_refine_sample_noop,
        observer=observer,
        num_samples=50,  # 50 attempts: every combination is drawn, bar a one-in-a-million run
        dimensions=[
            DecomposedDimension(
                dimension=Dimension(name="d1", description="d"),
                instantiations=[
                    Instantiation(description="i1"),
                    Instantiation(description="i2"),
                ],
            ),
            DecomposedDimension(
                dimension=Dimension(name="d2", description="d"),
                instantiations=[Instantiation(description="i3")],
            ),
        ],
    )

    # Don't care about the samples in this test
    async for _sample in run:
        pass

    pairs = {(r.dimension, r.instantiation) for r in observer.records}
    assert pairs.issubset({("d1", "i1"), ("d1", "i2"), ("d2", "i3")})
    assert len(pairs) == 3
