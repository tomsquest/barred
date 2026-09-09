from barred.llm import LLM
from barred.types import Criterion, Example, Instantiation, Sample


async def refine_sample(  # noqa: PLR0913
    llm: LLM,
    *,
    criterion: Criterion,
    example: Example,
    instantiation: Instantiation,
    sample: Sample,
    dissenting_feedback: list[str],
) -> Sample:
    """Refine a rejected sample using the dissenting judges' feedback.

    The configuration that produced it is kept: same dimension, same instantiation, same
    target label. Only the input block and its reasoning are rewritten, so the refined
    sample still covers the axis it was drawn for and can re-enter the debate on equal
    terms.

    Args:
        llm: Model used for every call.
        criterion: The property samples are labeled against.
        example: Unlabeled input block showing the domain, structure and style to stay
            within.
        instantiation: The concrete case the input block must keep exhibiting.
        sample: The rejected sample. Its label is carried over to the refined one.
        dissenting_feedback: Reasonings of the judges who rejected it, to be addressed.

    Returns:
        A new sample with the same label, and a rewritten input block and reasoning.

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
        context="refine_sample",
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
                    target_verdict=sample.label,
                    sample=sample,
                    dissenting_feedback=dissenting_feedback,
                ),
            },
        ],
    )

    refined = result.parsed

    # Preserve the target label y: refinement never changes it
    refined.label = sample.label

    llm.observer.on_sample_refined(
        criterion=criterion,
        example=example,
        instantiation=instantiation,
        previous=sample,
        refined=refined,
        dissenting_feedback=dissenting_feedback,
    )

    return refined


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

IMPORTANT: A previous attempt at generating an input block that fulfills the requirements A,B
failed verification, because the debaters could not agree that its label is <TARGET_VERDICT>.

Take a careful look at the previous generation under <FAILED_INPUT_BLOCK> and at the
arguments of the dissenting debaters under <DISSENTING_DEBATERS_ARGUMENTS>.
Then make another attempt at generating the input block so that:
- (i) the scenario aligns with the dimension under <DIMENSION>
- (ii) the input block gets a label of <TARGET_VERDICT> on the following <CRITERION>.

Please output the revised input block and the label you believe it should get for the CRITERION,
including reasoning. In your output, do not mention anything about the debaters and their arguments.
"""


def _get_user_message(  # noqa: PLR0913
    *,
    criterion: Criterion,
    example: Example,
    instantiation: Instantiation,
    target_verdict: bool,
    sample: Sample,
    dissenting_feedback: list[str],
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

<FAILED_INPUT_BLOCK>
{sample.input_block}
</FAILED_INPUT_BLOCK>

<DISSENTING_DEBATERS_ARGUMENTS>
{"\n".join(f"- {feedback}" for feedback in dissenting_feedback)}
</DISSENTING_DEBATERS_ARGUMENTS>
"""
