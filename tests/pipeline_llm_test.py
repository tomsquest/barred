import pytest

from barred.llm import LLM
from barred.pipeline import barred
from barred.types import DecomposedDimension, Dimension, Instantiation

pytestmark = pytest.mark.llm

CRITERION = "True if the sentence mentions a price in euros, False otherwise."
EXAMPLE = "Delivery costs €4.90, tax included."
DIMENSION = DecomposedDimension(
    dimension=Dimension(
        name="Register",
        description="How formal the wording is",
    ),
    instantiations=[
        Instantiation(description="A terse internal chat message between colleagues")
    ],
)


async def test_ok(llm: LLM) -> None:
    samples = [
        sample
        async for sample in barred(
            llm,
            criterion=CRITERION,
            examples=[EXAMPLE],
            dimensions=[DIMENSION],
            num_samples=1,
            max_attempts=3,
            max_debate_rounds=1,
            max_refine_rounds=1,
        )
    ]

    assert llm.total_usage.total_tokens > 0
    assert len(samples) == 1
    assert samples[0].reasoning
    assert samples[0].input_block
    assert samples[0].input_block != EXAMPLE
    assert isinstance(samples[0].label, bool)
