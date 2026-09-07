import pytest

from barred.llm import LLM
from barred.refine_sample import refine_sample
from barred.types import Instantiation, Sample

pytestmark = pytest.mark.llm


async def test_ok(llm: LLM) -> None:
    criterion = "True for a positive sentence, False otherwise."
    sample = Sample(
        reasoning="The sentence conveys subtle optimism",
        input_block="IT IS INCREDIBLE !!!",
        label=True,
    )

    refined_sample = await refine_sample(
        llm=llm,
        criterion=criterion,
        example="The sun is shining.",
        instantiation=Instantiation(
            description="A sentence expressing mild approval, polite pleasantries, or low-key optimism"
        ),
        sample=sample,
        dissenting_feedback=[
            "The sentence is too optimistic.",
            "The multiple exclamation marks are excessive.",
        ],
    )

    # A new reasoning and input block should be generated
    assert refined_sample.reasoning != sample.reasoning
    assert refined_sample.input_block != sample.input_block
    # The label should be the same
    assert refined_sample.label is sample.label
