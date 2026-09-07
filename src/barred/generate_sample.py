from barred.llm import LLM
from barred.types import Criterion, Example, Instantiation, Sample


async def generate_sample(
    llm: LLM,
    *,
    criterion: Criterion,
    example: Example,
    instantiation: Instantiation,
    target_verdict: bool,
) -> Sample:
    """Generate a fresh sample for `instantiation`, targeting `target_verdict` on `criterion`.

    The input block is written to sit near the decision boundary rather than deep inside
    it: a trivially labeled one teaches a classifier very little. `example` anchors the
    domain and style, and the model returns the reasoning behind the label it assigns,
    which later serves as the sample's case during the debate.

    Args:
        llm: Model used for every call.
        criterion: The property samples are labeled against.
        example: Unlabeled input block showing the domain, structure and style to stay
            within.
        instantiation: The concrete case the input block must exhibit.
        target_verdict: The label the input block should get on `criterion`. The model
            may still return a different one.

    Returns:
        The generated input block, its label and the reasoning behind it.

    Raises:
        LLMCallError: An LLM call exhausted its retries.
    """

    result = await llm.call(
        temperature=1.0,
        # No seed on purpose: identical prompts recur across the sample budget
        # (finite pool of examples x instantiations x verdicts), so a fixed seed would
        # collapse them into duplicate input blocks, defeating the pipeline's diversity goal.
        seed=None,
        response_format=Sample,
        context="generate_sample",
        messages=[
            {
                "role": "system",
                "content": _get_system_message(),
            },
            {
                "role": "user",
                "content": _get_user_message(
                    criterion=criterion,
                    example=example,
                    instantiation=instantiation,
                    target_verdict=target_verdict,
                ),
            },
        ],
    )

    sample = result.parsed
    llm.observer.on_sample_generated(
        criterion=criterion,
        example=example,
        instantiation=instantiation,
        target_verdict=target_verdict,
        sample=sample,
    )
    return sample


def _get_system_message() -> str:
    return """\
You are part of a test case generation system, whose goal is to create test cases for
testing a classifier given under <CRITERION>.
Each test case is composed of an input block and a label.

Your goal is to generate an input block that fulfills the following requirements:
A. it aligns with the dimension given under <DIMENSION>
B. the CRITERION's label is <TARGET_VERDICT> for the generated input block

Make sure the input block you create matches the domain and style of the <EXAMPLE_INPUT_BLOCK>.
You may use the same topic from <EXAMPLE_INPUT_BLOCK>, but you can diverge if this is required
for fulfilling the requirements above.
In your output, do not mention anything about test cases, models, dimensions or labels.

Aim to generate the input block as a challenging boundary case, rather than a trivial one.
At the end, it should be used to stress test a smart and successful classifier for the CRITERION.

When generating the input block, adhere as you can to the dimension given under <DIMENSION>.

Return the generated input block and the label you believe it should get for the CRITERION, including reasoning.
"""


def _get_user_message(
    *,
    criterion: Criterion,
    example: Example,
    instantiation: Instantiation,
    target_verdict: bool,
) -> str:
    return f"""\
<CRITERION>
{criterion}
</CRITERION>

<EXAMPLE_INPUT_BLOCK>
{example}
</EXAMPLE_INPUT_BLOCK>

<DIMENSION>
{instantiation.description}
</DIMENSION>

<TARGET_VERDICT>
{target_verdict}
</TARGET_VERDICT>
"""
