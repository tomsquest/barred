from types import SimpleNamespace
from typing import cast

from barred.debate import debate
from barred.llm import LLM
from barred.observer import NullObserver
from barred.types import JudgeVerdict, Sample

SAMPLE_LABELED_TRUE = Sample(
    reasoning="reasoning", input_block="input block", label=True
)
SAMPLE_LABELED_FALSE = Sample(
    reasoning="reasoning", input_block="input block", label=False
)


class _RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.rounds: list[tuple[int, bool]] = []
        self.final_round: int | None = None

    def on_debate_round(
        self,
        *,
        round_number: int,
        consensus: bool,
        **_: object,
    ) -> None:
        self.rounds.append((round_number, consensus))

    def on_debate_result(self, *, final_round: int, **_: object) -> None:
        self.final_round = final_round


class _FakeLLM:
    def __init__(self, rounds: list[list[bool]]) -> None:
        self.observer = _RecordingObserver()
        self.labels = [label for labels in rounds for label in labels]

    async def call(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            parsed=JudgeVerdict(
                reasoning="verdict reasoning",
                confidence="high",
                label=self.labels.pop(0),
            )
        )


async def test_judges_agree_on_the_first_round() -> None:
    llm = _FakeLLM([[True, True]])

    result = await debate(
        cast("LLM", llm), criterion="criterion", sample=SAMPLE_LABELED_TRUE
    )

    assert result.valid is True
    assert result.dissenting_feedback == []
    assert llm.observer.rounds == [(1, True)]
    assert llm.observer.final_round == 1


async def test_judges_agree_on_the_second_round() -> None:
    llm = _FakeLLM(
        [
            # disagree
            [True, False],
            # then agree
            [True, True],
        ]
    )

    result = await debate(
        cast("LLM", llm), criterion="criterion", sample=SAMPLE_LABELED_TRUE
    )

    assert result.valid is True
    assert result.dissenting_feedback == []
    assert llm.observer.rounds == [(1, False), (2, True)]
    assert llm.observer.final_round == 2


async def test_judges_agree_on_the_third_round() -> None:
    llm = _FakeLLM(
        [
            # disagree
            [True, False],
            # disagree again
            [True, False],
            # then agree
            [True, True],
        ]
    )

    result = await debate(
        cast("LLM", llm),
        criterion="criterion",
        sample=SAMPLE_LABELED_TRUE,
        max_debate_rounds=3,
    )

    assert result.valid is True
    assert result.dissenting_feedback == []
    assert llm.observer.rounds == [(1, False), (2, False), (3, True)]
    assert llm.observer.final_round == 3


async def test_judges_never_agree() -> None:
    llm = _FakeLLM(
        [
            # disagree
            [True, False],
            # then disagree
            [True, False],
        ]
    )

    result = await debate(
        cast("LLM", llm), criterion="criterion", sample=SAMPLE_LABELED_TRUE
    )

    assert result.valid is False
    assert result.dissenting_feedback == ["verdict reasoning"]
    assert llm.observer.rounds == [(1, False), (2, False)]
    assert llm.observer.final_round == 2


async def test_judges_agree_on_a_sample_labeled_false() -> None:
    # Consensus is on the sample's own label, not on True
    llm = _FakeLLM([[False, False]])

    result = await debate(
        cast("LLM", llm), criterion="criterion", sample=SAMPLE_LABELED_FALSE
    )

    assert result.valid is True
    assert result.dissenting_feedback == []
    assert llm.observer.rounds == [(1, True)]
    assert llm.observer.final_round == 1


async def test_judges_dissent_on_a_sample_labeled_false() -> None:
    llm = _FakeLLM([[True, True], [True, True]])

    result = await debate(
        cast("LLM", llm), criterion="criterion", sample=SAMPLE_LABELED_FALSE
    )

    assert result.valid is False
    assert result.dissenting_feedback == ["verdict reasoning", "verdict reasoning"]
    assert llm.observer.rounds == [(1, False), (2, False)]
    assert llm.observer.final_round == 2


async def test_stops_after_one_round_when_max_debate_rounds_is_one() -> None:
    # A single round to draw from: a second one would exhaust the fake and raise
    llm = _FakeLLM([[True, False]])

    result = await debate(
        cast("LLM", llm),
        criterion="criterion",
        sample=SAMPLE_LABELED_TRUE,
        max_debate_rounds=1,
    )

    assert result.valid is False
    assert result.dissenting_feedback == ["verdict reasoning"]
    assert llm.observer.rounds == [(1, False)]
    assert llm.observer.final_round == 1


async def test_stops_as_soon_as_judges_agree() -> None:
    llm = _FakeLLM([[True, True]])

    result = await debate(
        cast("LLM", llm),
        criterion="criterion",
        sample=SAMPLE_LABELED_TRUE,
        max_debate_rounds=5,
    )

    assert result.valid is True
    assert llm.observer.rounds == [(1, True)]
    assert llm.observer.final_round == 1
    assert llm.labels == []  # no verdict consumed beyond the first round
