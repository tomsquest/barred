import pytest

from barred.generate_sample import generate_sample
from barred.llm import LLM
from barred.types import Instantiation

pytestmark = pytest.mark.llm


async def test_ok(llm: LLM) -> None:
    criterion = "True for a positive sentence, False otherwise."
    example = "The sun is shining."
    instantiation = Instantiation(
        description="A sentence expressing mild approval, polite pleasantries, or low-key optimism"
    )
    target_verdict = True

    sample = await generate_sample(
        llm=llm,
        criterion=criterion,
        example=example,
        instantiation=instantiation,
        target_verdict=target_verdict,
    )

    assert sample.reasoning
    assert sample.input_block
    assert sample.label is target_verdict  # the LLM should respect the target verdict
