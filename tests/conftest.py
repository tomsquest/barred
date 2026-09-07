import pytest
from any_llm import LLMProvider
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.pretty import pprint

from barred.llm import LLM
from barred.observer import Message, NullObserver
from barred.types import (
    DebateResult,
    DecomposedDimension,
    Dimension,
    GenerationRecord,
    Instantiation,
    JudgeVerdict,
    Sample,
)

load_dotenv()


class PrintEventsObserver(NullObserver):
    def on_dimensions_generated(
        self,
        *,
        criterion: str,  # noqa: ARG002
        examples: list[str],  # noqa: ARG002
        dimensions: list[Dimension],
    ) -> None:
        print(f"\n{'=' * 20} Dimensions Generated {'=' * 20}")
        for dimension in dimensions:
            print(f"- {dimension.name}: {dimension.description}")

    def on_dimensions_deduplicated(
        self,
        *,
        dimensions: list[Dimension],
    ) -> None:
        print(f"\n{'=' * 20} Dimensions Deduplicated {'=' * 20}")
        for dimension in dimensions:
            print(f"- {dimension.name}: {dimension.description}")

    def on_instantiations_generated(
        self,
        *,
        dimension: Dimension,
        instantiations: list[Instantiation],
    ) -> None:
        print(f"\n{'=' * 20} Instantiations for {dimension.name!r} {'=' * 20}")
        for instantiation in instantiations:
            print(f"- {instantiation.description}")

    def on_dimensions_decomposed(
        self,
        *,
        criterion: str,  # noqa: ARG002
        examples: list[str],  # noqa: ARG002
        dimensions: list[DecomposedDimension],
    ) -> None:
        print(f"\n{'=' * 20} Decomposed ({len(dimensions)} dimensions) {'=' * 20}")

    def on_attempt_drawn(
        self,
        *,
        attempt: int,
        dimension: Dimension,
        instantiation: Instantiation,
        example: str,
        target_verdict: bool,
    ) -> None:
        print(f"\n{'=' * 20} Attempt {attempt} Drawn {'=' * 20}")
        print(
            f"dim={dimension.name!r} inst={instantiation.description!r} "
            f"example={example!r} target_verdict={target_verdict}"
        )

    def on_sample_generated(
        self,
        *,
        criterion: str,  # noqa: ARG002
        example: str,  # noqa: ARG002
        instantiation: Instantiation,  # noqa: ARG002
        target_verdict: bool,
        sample: Sample,
    ) -> None:
        print(f"\n{'=' * 20} Sample Generated (target={target_verdict}) {'=' * 20}")
        pprint(sample)

    def on_debate_round(
        self,
        *,
        round_number: int,
        verdicts: list[JudgeVerdict],
        consensus: bool,
    ) -> None:
        print(f"\n{'=' * 20} Debate Round {round_number} {'=' * 20}")
        print(f"consensus={consensus}, {len(verdicts)} verdicts:")
        for verdict in verdicts:
            print(f"- {verdict=}")

    def on_debate_result(
        self,
        *,
        criterion: str,  # noqa: ARG002
        sample: Sample,  # noqa: ARG002
        result: DebateResult,
        final_round: int,
    ) -> None:
        print(f"\n{'=' * 20} Debate Result {'=' * 20}")
        print(f"final_round={final_round} {result=}")

    def on_sample_refined(  # noqa: PLR0913
        self,
        *,
        criterion: str,  # noqa: ARG002
        example: str,  # noqa: ARG002
        instantiation: Instantiation,  # noqa: ARG002
        previous: Sample,
        refined: Sample,
        dissenting_feedback: list[str],
    ) -> None:
        print(f"\n{'=' * 20} Sample Refined {'=' * 20}")
        print(f"addressing {len(dissenting_feedback)} objection(s)")
        print("--- previous ---")
        pprint(previous)
        print("--- refined ---")
        pprint(refined)

    def on_sample_finalized(
        self,
        *,
        record: GenerationRecord,
    ) -> None:
        print(f"\n{'=' * 20} Sample Finalized {'=' * 20}")
        pprint(record)

    def on_llm_retry(
        self,
        *,
        context: str,
        attempt: int,
        max_attempts: int,
        error: Exception,
        delay: float,
    ) -> None:
        print(
            f"\n[{context}] LLM retry {attempt}/{max_attempts} after {error!r}, "
            f"waiting {delay:.1f}s"
        )

    def on_llm_call(
        self,
        *,
        context: str,
        messages: list[Message],
        response: BaseModel,
    ) -> None:
        print(f"\n{'=' * 20} {context} {'=' * 20}")
        for message in messages:
            role = message["role"] if isinstance(message, dict) else message.role
            content = (
                message.get("content") if isinstance(message, dict) else message.content
            )
            print(f"Role: {role}\n{content}")
        print("--- response ---")
        pprint(response)


@pytest.fixture
def observer() -> PrintEventsObserver:
    return PrintEventsObserver()


@pytest.fixture
def llm(observer: PrintEventsObserver) -> LLM:
    provider = LLMProvider.OPENROUTER
    model = "nvidia/nemotron-3-super-120b-a12b:free"

    return LLM(
        provider=provider,
        model=model,
        reasoning_effort="medium",
        max_tokens=16384,
        retry=2,
        retry_base_delay=2.0,
        observer=observer,
    )
