import pytest

from barred.debate import debate
from barred.llm import LLM
from barred.types import Sample

pytestmark = pytest.mark.llm

CRITERION = "True for a positive sentence, False otherwise."


async def test_consensus(llm: LLM) -> None:
    debate_result = await debate(
        llm=llm,
        criterion=CRITERION,
        sample=Sample(
            reasoning="The sentence states plain delight at the outcome",
            input_block="I'm delighted with how this turned out.",
            label=True,
        ),
    )

    assert debate_result.valid is True
    assert not debate_result.dissenting_feedback


async def test_no_consensus(llm: LLM) -> None:
    debate_result = await debate(
        llm=llm,
        criterion=CRITERION,
        sample=Sample(
            reasoning="The sentence is hostile, not positive",
            input_block="We hate each other.",
            label=True,
        ),
    )

    assert debate_result.valid is False
    assert debate_result.dissenting_feedback
