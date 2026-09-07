import pytest

from barred.decompose_dimensions import decompose_dimensions
from barred.llm import LLM

pytestmark = pytest.mark.llm


async def test_ok(llm: LLM) -> None:
    criterion = "True for a positive sentence, False otherwise."
    examples = [
        "The sun is shining.",
        "I love programming.",
        "You have a great idea!",
        "We hate each other.",
        "Today it's pouring.",
        "I am cold",
    ]

    dimensions = await decompose_dimensions(
        llm=llm,
        criterion=criterion,
        examples=examples,
    )

    assert len(dimensions)
    for d in dimensions:
        assert len(d.instantiations)
