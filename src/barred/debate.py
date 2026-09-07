import asyncio

from pydantic import BaseModel

from barred.llm import LLM
from barred.types import Criterion, DebateResult, JudgeVerdict, Sample


async def debate(
    llm: LLM,
    *,
    criterion: Criterion,
    sample: Sample,
    max_debate_rounds: int = 2,
) -> DebateResult:
    """Run an asymmetric debate over a sample to decide whether it is valid.

    The asymmetry is twofold. One judge favors recall, the other precision; and the case
    for the sample is the reasoning produced alongside it, replayed unchanged at every
    round, so it argues for the target label and never concedes. The judges first rule
    independently, then re-rule seeing that case and each other's verdicts.

    The sample is valid only once they converge on its target label. Otherwise, the
    dissenting reasonings come back as feedback for refinement.

    Rounds stop as soon as the judges agree, and at `max_debate_rounds` at the latest.

    Args:
        llm: Model used for every call.
        criterion: The property samples are labeled against.
        sample: The sample under debate; its own label is the one to converge on, and
            its reasoning is the case argued on its behalf.
        max_debate_rounds: Upper bound on debate rounds. Min 1.

    Returns:
        Whether the judges upheld the sample's label, with the reasonings of those who
        did not.

    Raises:
        ValueError: `max_debate_rounds` is not strictly positive.
        LLMCallError: An LLM call exhausted its retries.
    """
    if max_debate_rounds <= 0:
        msg = f"max_debate_rounds must be > 0, got {max_debate_rounds}"
        raise ValueError(msg)

    # First round, the judges evaluate the sample independently
    round_number = 1
    verdicts = await asyncio.gather(
        *[
            _ask_judge(
                llm,
                _get_first_round_message(
                    persona=persona,
                    criterion=criterion,
                    sample=sample,
                ),
                context=f"debate:round{round_number}:judge={persona.name}",
            )
            for persona in _JUDGE_PERSONAS
        ]
    )
    consensus = _is_consensus(verdicts, target_label=sample.label)
    llm.observer.on_debate_round(
        round_number=round_number, verdicts=verdicts, consensus=consensus
    )

    # Next rounds, the judges evaluate the sample with the sample's reasoning and the previous verdicts as input
    while not consensus and round_number < max_debate_rounds:
        round_number += 1
        verdicts = await asyncio.gather(
            *[
                _ask_judge(
                    llm,
                    _get_subsequent_debate_round_message(
                        persona=persona,
                        criterion=criterion,
                        sample=sample,
                        own_verdict=verdicts[i],
                        other_verdicts=[v for j, v in enumerate(verdicts) if j != i],
                    ),
                    context=f"debate:round{round_number}:judge={persona.name}",
                )
                for i, persona in enumerate(_JUDGE_PERSONAS)
            ]
        )
        consensus = _is_consensus(verdicts, target_label=sample.label)
        llm.observer.on_debate_round(
            round_number=round_number, verdicts=verdicts, consensus=consensus
        )

    result = DebateResult(
        valid=consensus,
        dissenting_feedback=[
            verdict.reasoning for verdict in verdicts if verdict.label != sample.label
        ],
    )
    llm.observer.on_debate_result(
        criterion=criterion,
        sample=sample,
        result=result,
        final_round=round_number,
    )
    return result


class _JudgePersona(BaseModel):
    name: str
    instructions: str


_JUDGE_PERSONAS: list[_JudgePersona] = [
    _JudgePersona(
        name="recall-focused",
        instructions=(
            "Prioritize recall: if there is a reasonable interpretation of the input block "
            "under which the CRITERION holds, lean towards True rather than dismissing it."
        ),
    ),
    _JudgePersona(
        name="strict",
        instructions=(
            "Be strict and favor precision, with no flexibility or interpretive leeway: "
            "only answer True when the input block unambiguously satisfies the CRITERION."
        ),
    ),
]


def _get_first_round_message(
    *,
    persona: _JudgePersona,
    criterion: Criterion,
    sample: Sample,
) -> str:
    return f"""\
You are a {persona.name} evaluation agent evaluating whether the given CRITERION holds on the given input block.
The answer should be True (CRITERION holds) or False (CRITERION doesn't hold).
{persona.instructions}

<CRITERION>
{criterion}
</CRITERION>

<INPUT_BLOCK>
{sample.input_block}
</INPUT_BLOCK>
"""


def _format_verdict(verdict: JudgeVerdict) -> str:
    return f"""\
- Reasoning: {verdict.reasoning}
- Label: {verdict.label}
- Confidence: {verdict.confidence}
"""


def _get_subsequent_debate_round_message(
    *,
    persona: _JudgePersona,
    criterion: Criterion,
    sample: Sample,
    own_verdict: JudgeVerdict,
    other_verdicts: list[JudgeVerdict],
) -> str:
    agents_responses = f"""\
Advocate:
- Reasoning: {sample.reasoning}
- Label: {sample.label}
{"\n\n".join(f"Judge:\n{_format_verdict(verdict)}" for verdict in other_verdicts)}
"""

    return f"""\
You are participating in a multi-agent debate for a classification task evaluating
whether the given CRITERION holds on the given input block.
The answer should be True (CRITERION holds) or False (CRITERION doesn't hold).
{persona.instructions}

<CRITERION>
{criterion}
</CRITERION>

<INPUT_BLOCK>
{sample.input_block}
</INPUT_BLOCK>

Your Previous Response:
{_format_verdict(own_verdict)}

Other Agents' Responses:
{agents_responses}

Arriving here means that at least one of the other debate agents did not agree with your previous label.
Therefore, please carefully examine your previous response and the other agents' responses.
This doesn't mean you should automatically accept their arguments and switch the label.
Instead, read through their arguments and determine if they are both correct and relevant,
and determine which label the given CRITERION assigns to the given input block.
Provide an updated response with your final classification.
"""


async def _ask_judge(llm: LLM, message: str, *, context: str) -> JudgeVerdict:
    result = await llm.call(
        temperature=0.0,  # Determinist
        seed=0,
        response_format=JudgeVerdict,
        context=context,
        messages=[
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    return result.parsed


def _is_consensus(verdicts: list[JudgeVerdict], *, target_label: bool) -> bool:
    return all(verdict.label == target_label for verdict in verdicts)
