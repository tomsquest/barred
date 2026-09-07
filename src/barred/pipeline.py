import asyncio
import secrets
from collections.abc import AsyncGenerator, Sequence

from barred.debate import debate
from barred.generate_sample import generate_sample
from barred.llm import LLM
from barred.refine_sample import refine_sample
from barred.types import (
    Criterion,
    DecomposedDimension,
    Example,
    GenerationRecord,
    Instantiation,
    RefinementRound,
    Sample,
)


def barred(  # noqa: C901, PLR0913
    llm: LLM,
    *,
    criterion: Criterion,
    examples: list[Example],
    dimensions: list[DecomposedDimension],
    num_samples: int,
    concurrency: int = 8,
    max_debate_rounds: int = 2,
    max_refine_rounds: int = 2,
    max_attempts: int | None = None,
) -> AsyncGenerator[Sample]:
    """Run the draw -> generate -> debate -> refine loop and stream the samples.

    Step 2 of the pipeline. Each attempt draws a dimension, one of its instantiations and
    an example, generates a sample from them, then debates it; a sample the debate rejects
    is refined and debated again, up to `max_refine_rounds` times. Each debate itself runs
    up to `max_debate_rounds` rounds of judges.

    Args:
        llm: Model used for every call.
        criterion: The property samples are labeled against.
        examples: Unlabeled input blocks showing the domain, structure and style to
            stay within.
        dimensions: Axes to draw from, as returned by `decompose_dimensions`.
        num_samples: Number of accepted samples to reach before stopping.
        concurrency: Attempts running in parallel (1 to watch the run step by step).
        max_debate_rounds: Rounds of judges allowed per debate, before or after a refinement.
        max_refine_rounds: Refinement passes allowed per attempt after a rejected debate.
        max_attempts: Attempts allowed before the run stops, reached or not. Defaults to
            `num_samples * 2`, so a run costs at worst twice a perfect one.

    Yields:
        One `Sample` per accepted attempt, as it lands. Fewer than `num_samples` when
        `max_attempts` runs out first.

    Raises:
        ValueError: An argument is empty or out of range.
        LLMCallError: A call exhausted its retries, which ends the run: `llm.call` already
            retried whatever was worth retrying.
    """

    if not criterion:
        msg = "criterion must not be empty"
        raise ValueError(msg)
    if not examples:
        msg = "examples must not be empty"
        raise ValueError(msg)
    if not dimensions:
        msg = "dimensions must not be empty"
        raise ValueError(msg)
    for decomposed in dimensions:
        if not decomposed.instantiations:
            msg = f"dimension {decomposed.dimension.name!r} has no instantiations"
            raise ValueError(msg)
    if num_samples < 1:
        msg = "num_samples must be >= 1"
        raise ValueError(msg)
    if concurrency < 1:
        msg = "concurrency must be >= 1"
        raise ValueError(msg)
    if max_debate_rounds < 1:
        msg = "max_debate_rounds must be >= 1"
        raise ValueError(msg)
    if max_refine_rounds < 0:
        msg = "max_refine_rounds must be >= 0"
        raise ValueError(msg)
    if max_attempts is not None and max_attempts < 1:
        msg = "max_attempts must be >= 1"
        raise ValueError(msg)

    return _run(
        llm,
        criterion=criterion,
        examples=examples,
        dimensions=dimensions,
        num_samples=num_samples,
        concurrency=concurrency,
        max_debate_rounds=max_debate_rounds,
        max_refine_rounds=max_refine_rounds,
        max_attempts=num_samples * 2 if max_attempts is None else max_attempts,
    )


async def _run(  # noqa: PLR0913
    llm: LLM,
    *,
    criterion: Criterion,
    examples: list[Example],
    dimensions: list[DecomposedDimension],
    num_samples: int,
    concurrency: int,
    max_debate_rounds: int,
    max_refine_rounds: int,
    max_attempts: int,
) -> AsyncGenerator[Sample]:
    """Keep `concurrency` attempts in flight, until `num_samples` are accepted or `max_attempts` spent.

    Split from `barred` so the argument checks raise on call rather than on the first iteration,
    as a bare generator would.
    """

    remaining = num_samples  # acceptances still owed
    attempted = 0
    tasks: set[asyncio.Task[GenerationRecord]] = set()

    try:
        while (remaining and attempted < max_attempts) or tasks:
            while len(tasks) < min(concurrency, remaining, max_attempts - attempted):
                attempted += 1
                tasks.add(
                    asyncio.create_task(
                        _attempt(
                            llm,
                            attempt_number=attempted,
                            criterion=criterion,
                            dimensions=dimensions,
                            examples=examples,
                            max_debate_rounds=max_debate_rounds,
                            max_refine_rounds=max_refine_rounds,
                        )
                    )
                )

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Retrieve every exception before raising any of them: the whole batch tends
            # to fail together (one bad key fails all the calls in flight), and asyncio
            # dumps the traceback of each unretrieved one over the error actually raised.
            outcomes = [(task, task.exception()) for task in done]
            for task, error in outcomes:
                if error is not None:
                    raise error
                record = task.result()
                if record.accepted:
                    yield record.final_sample
                    remaining -= 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _attempt(  # noqa: PLR0913
    llm: LLM,
    *,
    criterion: Criterion,
    examples: list[Example],
    dimensions: list[DecomposedDimension],
    attempt_number: int,
    max_debate_rounds: int,
    max_refine_rounds: int,
) -> GenerationRecord:
    dimension = _draw(dimensions)
    instantiation = _draw(dimension.instantiations)
    example = _draw(examples)
    target_verdict = secrets.choice((True, False))

    llm.observer.on_attempt_drawn(
        attempt=attempt_number,
        dimension=dimension.dimension,
        instantiation=instantiation,
        example=example,
        target_verdict=target_verdict,
    )

    sample = await generate_sample(
        llm=llm,
        criterion=criterion,
        example=example,
        instantiation=instantiation,
        target_verdict=target_verdict,
    )

    accepted, refinement_rounds = await _debate_and_refine(
        llm,
        criterion=criterion,
        instantiation=instantiation,
        example=example,
        sample=sample,
        max_debate_rounds=max_debate_rounds,
        max_refine_rounds=max_refine_rounds,
    )

    record = GenerationRecord(
        attempt=attempt_number,
        dimension=dimension.dimension.name,
        instantiation=instantiation.description,
        example=example,
        target_verdict=target_verdict,
        accepted=accepted,
        refinement_rounds=refinement_rounds,
        final_sample=refinement_rounds[-1].sample,
    )

    llm.observer.on_sample_finalized(record=record)
    return record


async def _debate_and_refine(  # noqa: PLR0913
    llm: LLM,
    *,
    criterion: Criterion,
    example: Example,
    instantiation: Instantiation,
    sample: Sample,
    max_debate_rounds: int,
    max_refine_rounds: int,
) -> tuple[bool, list[RefinementRound]]:
    """Debate `sample`, refining and re-debating it while the judges reject it.

    Returns whether it was ever accepted, and every round it went through.
    """

    refinement_rounds: list[RefinementRound] = []
    for round_i in range(max_refine_rounds + 1):
        debate_result = await debate(
            llm=llm,
            criterion=criterion,
            sample=sample,
            max_debate_rounds=max_debate_rounds,
        )
        refinement_rounds.append(
            RefinementRound(
                sample=sample,
                valid=debate_result.valid,
                dissenting_feedback=debate_result.dissenting_feedback,
            )
        )
        if debate_result.valid:
            return True, refinement_rounds

        # On the final turn, refining again would yield a sample we never validate.
        if round_i == max_refine_rounds:
            break

        sample = await refine_sample(
            llm=llm,
            criterion=criterion,
            example=example,
            instantiation=instantiation,
            sample=sample,
            dissenting_feedback=debate_result.dissenting_feedback,
        )

    return False, refinement_rounds


def _draw[T](sequence: Sequence[T]) -> T:
    return secrets.choice(sequence)
