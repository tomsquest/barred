from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from barred.logger import logger
from barred.types import (
    Criterion,
    DebateResult,
    DecomposedDimension,
    Dimension,
    Example,
    GenerationRecord,
    Instantiation,
    JudgeVerdict,
    Message,
    Sample,
)


@runtime_checkable
class Observer(Protocol):
    """Receives events emitted by the pipeline components.

    This is the only window into a run, which is otherwise silent. Implement it to
    persist records, drive a progress display or collect metrics. `LoggingObserver`
    covers the usual need; the default, `NullObserver`, drops everything.
    """

    def on_dimensions_generated(
        self,
        *,
        criterion: Criterion,
        examples: list[Example],
        dimensions: list[Dimension],
    ) -> None:
        """`decompose_dimensions` produced the raw (not yet deduplicated) candidate dimensions."""
        ...

    def on_dimensions_deduplicated(
        self,
        *,
        dimensions: list[Dimension],
    ) -> None:
        """`decompose_dimensions` merged the candidate dimensions into a deduplicated list."""
        ...

    def on_instantiations_generated(
        self,
        *,
        dimension: Dimension,
        instantiations: list[Instantiation],
    ) -> None:
        """`decompose_dimensions` produced the instantiations for one dimension."""
        ...

    def on_dimensions_decomposed(
        self,
        *,
        criterion: Criterion,
        examples: list[Example],
        dimensions: list[DecomposedDimension],
    ) -> None:
        """`decompose_dimensions` finished: deduplicated dimensions with their instantiations."""
        ...

    def on_attempt_drawn(
        self,
        *,
        attempt: int,
        dimension: Dimension,
        instantiation: Instantiation,
        example: Example,
        target_verdict: bool,
    ) -> None:
        """`barred` started an attempt from a fresh random draw.

        `attempt` is the 1-based ordinal of the attempt across the run.
        """
        ...

    def on_sample_generated(
        self,
        *,
        criterion: Criterion,
        example: Example,
        instantiation: Instantiation,
        target_verdict: bool,
        sample: Sample,
    ) -> None:
        """A fresh sample was produced by `generate_sample`."""
        ...

    def on_debate_round(
        self,
        *,
        round_number: int,
        verdicts: list[JudgeVerdict],
        consensus: bool,
    ) -> None:
        """One debate round completed."""
        ...

    def on_debate_result(
        self,
        *,
        criterion: Criterion,
        sample: Sample,
        result: DebateResult,
        final_round: int,
    ) -> None:
        """A debate concluded (accepted or rejected) on `sample` at `final_round`."""
        ...

    def on_sample_refined(  # noqa: PLR0913
        self,
        *,
        criterion: Criterion,
        example: Example,
        instantiation: Instantiation,
        previous: Sample,
        refined: Sample,
        dissenting_feedback: list[str],
    ) -> None:
        """A rejected sample was rewritten by `refine_sample` to address the objections."""
        ...

    def on_sample_finalized(
        self,
        *,
        record: GenerationRecord,
    ) -> None:
        """`barred` finished an attempt (accepted or discarded)."""
        ...

    def on_llm_retry(
        self,
        *,
        context: str,
        attempt: int,
        max_attempts: int,
        error: Exception,
        delay: float,
    ) -> None:
        """A transient LLM failure occurred; the call will be retried after `delay` seconds.

        `attempt` is 1-based and counts the attempt that just failed.
        """
        ...

    def on_llm_call(
        self,
        *,
        context: str,
        messages: list[Message],
        response: BaseModel,
    ) -> None:
        """One LLM call completed successfully."""
        ...


class NullObserver:
    """Default observer that ignores every event."""

    def on_dimensions_generated(
        self,
        *,
        criterion: Criterion,
        examples: list[Example],
        dimensions: list[Dimension],
    ) -> None:
        pass

    def on_dimensions_deduplicated(
        self,
        *,
        dimensions: list[Dimension],
    ) -> None:
        pass

    def on_instantiations_generated(
        self,
        *,
        dimension: Dimension,
        instantiations: list[Instantiation],
    ) -> None:
        pass

    def on_dimensions_decomposed(
        self,
        *,
        criterion: Criterion,
        examples: list[Example],
        dimensions: list[DecomposedDimension],
    ) -> None:
        pass

    def on_attempt_drawn(
        self,
        *,
        attempt: int,
        dimension: Dimension,
        instantiation: Instantiation,
        example: Example,
        target_verdict: bool,
    ) -> None:
        pass

    def on_sample_generated(
        self,
        *,
        criterion: Criterion,
        example: Example,
        instantiation: Instantiation,
        target_verdict: bool,
        sample: Sample,
    ) -> None:
        pass

    def on_debate_round(
        self,
        *,
        round_number: int,
        verdicts: list[JudgeVerdict],
        consensus: bool,
    ) -> None:
        pass

    def on_debate_result(
        self,
        *,
        criterion: Criterion,
        sample: Sample,
        result: DebateResult,
        final_round: int,
    ) -> None:
        pass

    def on_sample_refined(  # noqa: PLR0913
        self,
        *,
        criterion: Criterion,
        example: Example,
        instantiation: Instantiation,
        previous: Sample,
        refined: Sample,
        dissenting_feedback: list[str],
    ) -> None:
        pass

    def on_sample_finalized(
        self,
        *,
        record: GenerationRecord,
    ) -> None:
        pass

    def on_llm_call(
        self,
        *,
        context: str,
        messages: list[Message],
        response: BaseModel,
    ) -> None:
        pass

    def on_llm_retry(
        self,
        *,
        context: str,
        attempt: int,
        max_attempts: int,
        error: Exception,
        delay: float,
    ) -> None:
        pass


class LoggingObserver(NullObserver):
    """Observer that logs pipeline events, so a run can be audited from the logs.

    Prompts and parsed responses go to DEBUG (verbose, only shown when the log level allows);
    business milestones go to INFO.
    """

    def on_dimensions_generated(
        self,
        *,
        criterion: Criterion,  # noqa: ARG002
        examples: list[Example],  # noqa: ARG002
        dimensions: list[Dimension],
    ) -> None:
        logger.info(
            f"[decompose_dimensions] generated {len(dimensions)} raw dimension(s)"
        )

    def on_dimensions_deduplicated(
        self,
        *,
        dimensions: list[Dimension],
    ) -> None:
        logger.info(
            f"[decompose_dimensions] deduplicated to {len(dimensions)} dimension(s)"
        )

    def on_instantiations_generated(
        self,
        *,
        dimension: Dimension,
        instantiations: list[Instantiation],
    ) -> None:
        logger.info(
            f"[decompose_dimensions] {len(instantiations)} instantiation(s) "
            f"for dimension {dimension.name!r}"
        )

    def on_dimensions_decomposed(
        self,
        *,
        criterion: Criterion,  # noqa: ARG002
        examples: list[Example],  # noqa: ARG002
        dimensions: list[DecomposedDimension],
    ) -> None:
        logger.info(
            f"[decompose_dimensions] Decomposed in {len(dimensions)} dimension(s)"
        )
        for i, decomposed in enumerate(dimensions, 1):
            logger.debug(
                f"{i}. {decomposed.dimension.name} ({len(decomposed.instantiations)} instantiation(s))"
            )
            for j, instantiation in enumerate(decomposed.instantiations, 1):
                logger.debug(f"  {i}.{j}. {instantiation.description}")

    def on_attempt_drawn(
        self,
        *,
        example: Example,  # noqa: ARG002
        attempt: int,
        dimension: Dimension,
        instantiation: Instantiation,
        target_verdict: bool,
    ) -> None:
        logger.info(
            f"Attempt {attempt} - Draw: dim={dimension.name!r} "
            f"inst={instantiation.description!r} verdict={target_verdict}"
        )

    def on_sample_generated(
        self,
        *,
        criterion: Criterion,  # noqa: ARG002
        example: Example,  # noqa: ARG002
        instantiation: Instantiation,
        target_verdict: bool,
        sample: Sample,
    ) -> None:
        logger.debug(
            f"[generate_sample] target={target_verdict} "
            f"inst={instantiation.description!r} -> "
            f"{sample.input_block!r} (label={sample.label})"
        )
        if sample.label != target_verdict:
            logger.warning(
                f"[generate_sample] generated label {sample.label} "
                f"instead of the target verdict {target_verdict}"
            )

    def on_debate_round(
        self,
        *,
        round_number: int,
        verdicts: list[JudgeVerdict],
        consensus: bool,
    ) -> None:
        logger.debug(f"[debate:round{round_number}] consensus={consensus}")
        for i, verdict in enumerate(verdicts, 1):
            logger.debug(
                f"[debate:round{round_number}:judge{i}] label={verdict.label} "
                f"confidence={verdict.confidence} - {verdict.reasoning}"
            )

    def on_debate_result(
        self,
        *,
        criterion: Criterion,  # noqa: ARG002
        sample: Sample,
        result: DebateResult,
        final_round: int,
    ) -> None:
        logger.debug(
            f"[debate] {sample.input_block!r} valid={result.valid} "
            f"final_round={final_round} dissenting={len(result.dissenting_feedback)}"
        )

    def on_sample_refined(  # noqa: PLR0913
        self,
        *,
        criterion: Criterion,  # noqa: ARG002
        example: Example,  # noqa: ARG002
        instantiation: Instantiation,
        previous: Sample,
        refined: Sample,
        dissenting_feedback: list[str],
    ) -> None:
        logger.debug(
            f"[refine_sample] inst={instantiation.description!r} "
            f"{previous.input_block!r} -> {refined.input_block!r} "
            f"(addressing {len(dissenting_feedback)} objection(s))"
        )

    def on_sample_finalized(
        self,
        *,
        record: GenerationRecord,
    ) -> None:
        logger.info(
            f"Attempt {record.attempt} "
            f"{'accepted' if record.accepted else 'discarded'} "
            f"after {len(record.refinement_rounds)} refinement round(s)"
        )

    def on_llm_call(
        self,
        *,
        context: str,
        messages: list[Message],
        response: BaseModel,
    ) -> None:
        logger.debug(f"[{context}] LLM call prompt:")
        for message in messages:
            role = message["role"] if isinstance(message, dict) else message.role
            content = (
                message.get("content") if isinstance(message, dict) else message.content
            )
            logger.debug(f"{role}:\n{content}")
        logger.debug(f"[{context}] LLM response: {response!r}")

    def on_llm_retry(
        self,
        *,
        context: str,
        attempt: int,
        max_attempts: int,
        error: Exception,
        delay: float,
    ) -> None:
        logger.warning(
            f"[{context}] LLM call failed (attempt {attempt}/{max_attempts}): "
            f"{error}. Retrying in {delay:.1f}s..."
        )
