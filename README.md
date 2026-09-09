<p align="center">
<img src="https://raw.githubusercontent.com/tomsquest/barred/main/doc/cover.png" alt="Cover" />
</p>

# BARRED: generate faithful and diverse training sets

[![PyPI](https://img.shields.io/pypi/v/barred?color=blue)](https://pypi.org/project/barred/)
[![Python versions](https://img.shields.io/pypi/pyversions/barred)](https://pypi.org/project/barred/)
[![License](https://img.shields.io/github/license/tomsquest/barred?color=green)](https://github.com/tomsquest/barred/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.25203-b31b1b)](https://arxiv.org/abs/2604.25203)

**Unofficial** implementation of the BARRED paper:

> Boundary Alignment Refinement through REflection and Debate, aka BARRED
> [arXiv:2604.25203](https://arxiv.org/pdf/2604.25203)

> [!NOTE]
> I developed this library for my own needs without endorsement. 
> It is not affiliated with the authors of the paper.

## What is BARRED?

In a nutshell: BARRED is a framework for generating **faithful** and **diverse** synthetic training data using only
a task description and a small set of unlabeled examples.

**TL;DR:** describe a task, get an annotated dataset

### Step by step

![Two steps](https://raw.githubusercontent.com/tomsquest/barred/main/doc/two_steps.png)

1. Define the task: a Criterion, plus a few examples of input you work on
    ```
    Criterion: True when the sentence expresses a positive sentiment, False otherwise

    Examples:
      - The delivery arrived two days late and the box was crushed.
      - Honestly, one of the best purchases I've made this year.
      - It works, I guess.
    ```
2. Let the library decompose the problem into dimensions
3. Then the library generates the training set

Results: a set of **labeled** samples:
```
1. The screws stripped on the first turn, total waste.  -> False
2. Arrived a day early, and better than the photos.     -> True
3. Does the job. Nothing more to say.                   -> False   <- the gray zone
```

## Why is it cool?

Over the past few years, here's what kept happening to me:
- I developed a classifier but got no dataset (business people were busy)
- I asked for relevancy judgments on products, but got quite nothing
- I begged for annotated samples of search queries but got none

So, BARRED is cool because it solves these problems by generating samples almost automatically.

And there is more. "Happy path" examples are easy to write. "Edge cases" are not, the gray zone, the "not true, not false".

Say you want to blacklist irrelevant products in search results (personal true story).
Anyone can tell that a hammer drill is relevant for a query on drills.
But what about drill bits? And a drill bit adapter? And a drill toy? And the dozen cases nobody thinks about?

That's where BARRED shines!

### Limits

Both come from following the paper closely, not from anything being hard to do.

1. **Binary classification only**: the label is `True` or `False`, no multi-class (e.g. `A/B/C`).  
   The code and the prompts could be adapted, I'm pretty confident. But paper=binary, library=binary.
2. **Plain text in, plain text out**: examples and generated samples are strings, no JSON schema/Structured output.  
   In one of my tests, the seed examples were `query + product title` pairs flattened into a string,
   so I had to parse the generated samples back to get the pieces.
   Passing a schema to `barred()` would have been possible. (that's for v2)

## Installation

```
uv add barred

# with Pip
pip install barred
```

This library builds on [any-llm from Mozilla AI](https://docs.mozilla.ai/any-llm/), and ships **no provider SDK by default**.  
You need the `any-llm-sdk` extra for the provider you want:

| Provider      | Install                                       |
|---------------|-----------------------------------------------|
| OpenAI        | `uv add barred "any-llm-sdk[openai]"`         |
| Anthropic     | `uv add barred "any-llm-sdk[anthropic]"`      |
| Gemini        | `uv add barred "any-llm-sdk[gemini]"`         |
| All providers | `uv add barred "any-llm-sdk[all]"`            |

See the [full list of providers](https://docs.mozilla.ai/providers).

Then set the matching key in your environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) or pass it to the `llm()` constructor.

Importing `barred` sets `ANY_LLM_UNIFIED_EXCEPTIONS=1` in your environment: the retry policy is written against
any-llm's unified exceptions, which are not raised by default yet
([any-llm#1369](https://github.com/mozilla-ai/any-llm/issues/1369)).

## Quick start

```python
import asyncio

from barred import LLM, barred, decompose_dimensions

#
# Step 1. Task definition
#
criterion = "True when the sentence expresses a positive sentiment, False otherwise"
examples = [
    "The delivery arrived two days late and the box was crushed.",
    "Honestly, one of the best purchases I've made this year.",
    "It works, I guess.",
]


async def main():
    # Your provider, your model.
    # The key is read from the environment (OPENAI_API_KEY here), or with the ` api_key ` param.
    llm = LLM(provider="openai", model="gpt-5.6-terra")

    #
    # Step 2. Decompose the Criterion into Dimensions
    #
    dimensions = await decompose_dimensions(llm, criterion=criterion, examples=examples)

    #
    # Step 3. Generate Samples, streamed as they land
    #
    async for sample in barred(
        llm,
        criterion=criterion,
        examples=examples,
        dimensions=dimensions,
        num_samples=5,
    ):
        print(sample.label, "->", sample.input_block)


asyncio.run(main())
```

### Parameters

The knobs you are likely to turn:

| Parameter           | Default           | What it does                                                                                                                      |
|---------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `num_samples`       | required          | Accepted samples to reach before stopping                                                                                         |
| `max_attempts`      | `2 × num_samples` | Attempt budget. The run stops there, reached or not, so a run costs at worst twice a perfect one                                  |
| `concurrency`       | `8`               | Attempts in flight. Set it to `1` to watch the run step by step                                                                   |
| `max_debate_rounds` | `2`               | Judge rounds per debate. `2` is the paper's `T`                                                                                   |
| `max_refine_rounds` | `2`               | Refinement passes per attempt. The paper does not give its `R_max`                                                                |
| `observer`          | none              | Set on `LLM(...)`. Gets every event, including the **rejected** samples and the token usage. `LoggingObserver` logs the whole run |

`decompose_dimensions()` has its own `concurrency` (default `5`).  
`LLM(...)` carries the model settings: `reasoning_effort` (default `"medium"`, the paper's setting), and the retry policy.

## Notebook/Demo

> [!NOTE]
> The notebook is the best way to **see BARRED in action**.
>
> It includes: dimensions decomposition and the generation of samples.  
> You will also be able to give a grasp on the generated data
>
> [Open Notebook in GitHub](https://github.com/tomsquest/barred/blob/main/notebooks/demo_sentiment_analysis.ipynb)  
> [Open Notebook in Google Colab](https://colab.research.google.com/github/tomsquest/barred/blob/main/notebooks/demo_sentiment_analysis.ipynb)

## How it works

The naive way is to let an LLM generate samples straight from the problem description. **WRONG**!

LLMs forget parts of the problem. They don't "explore" much, they stick to their few default answers.

### Dimensions and instantiations

The first thing that BARRED does is to decompose the problem into "Dimensions" and "Instantiations" of those dimensions. 

_I didn't know what an "instantiation" was (except in computing), but an `instantiation` is "concrete evidence in support of a concept/claim"._

Then, each instantiation will serve as a "seed" for the LLM to generate samples.

But can we generate samples directly from those instantiations right away? Of course not, that would be too simple!

### Debate and refinement

![Generating a sample](https://raw.githubusercontent.com/tomsquest/barred/main/doc/generate_sample.png)

BARRED makes two judges debate each sample until they agree.
When a judge disagrees, the sample is reworked using their feedback, then debated again, and so on. 

In the end, the generated samples can be accepted or not. This library only streams accepted samples, but you can also access the rejected ones using an `Observer`.

### The algorithm, in pseudocode

BARRED, as implemented:

```
1: Input: criterion C, examples E, target size N,
2:        concurrency K, max_debate_rounds B, max_refine_rounds R, max_attempts M = 2N
3:
4: # Step 1 — dimensions and their instantiations (paper lines 2-5, fused)
5: D ← decompose_dimensions(llm, criterion=C, examples=E, concurrency=5)
6:     # each d ∈ D is a DecomposedDimension(dimension, instantiations)
7:
8: # Step 2 — draw → generate → debate → refine, K attempts in flight
9: G ← ∅ ; attempts ← 0
10: while |G| < N and attempts < M do
11:     attempts ← attempts + 1
12:     d ~ Uniform(D)  ;  v ~ Uniform(d.instantiations)
13:     e ~ Uniform(E)  ;  y ~ Uniform({True, False})
14:
15:     s ← generate_sample(llm, criterion=C, example=e,
16:                         instantiation=v, target_verdict=y)
17:         # s is a Sample(reasoning, input_block, label)
18:
19:     for ℓ = 0, 1, ..., R do
20:         res ← debate(llm, criterion=C, sample=s, max_debate_rounds=B)
21:             # res is a DebateResult(valid, dissenting_feedback)
22:
23:         if res.valid then
24:             G ← G ∪ {s} ; yield s
25:             break
26:         end if
27:         if ℓ = R then break end if      # refining now would skip validation
28:
29:         s ← refine_sample(llm, criterion=C, example=e, instantiation=v,
30:                           sample=s, dissenting_feedback=res.dissenting_feedback)
31:     end for
32: end while
33: return G
```

### The algorithm, in bullets

(Because I like bullets, plus some notes)

1. Task definition
   1. Criterion = task description = the classification Criterion.  
      E.g., “Is the product relevant for the query?”
   2. Unlabeled examples  
      10 to 30 are enough  
      No label is needed, represent the “shape” of the input data.  
      E.g., “query: drill, product: bosch hammer drill”  
2. Dimensions decomposition
   1. dimension extraction
   2. dimension deduplication // it may happen, at least the Authors added that step
   3. dimension instantiation  
      ```
      For each dimension:
        instantiations = verbalized_sampling(dimension)
        // list of name + polarity + score
      ```
3. Sample Generation
   1. Take a dimension  
      Take an instantiation of this dimension  
      Take an example  
      Take a label (true/false)  
      (not random, take each in turn, aka uniform distribution)  
      Consequence:
        - binary label → 50/50 dataset → maybe not your reality  
        - The “polarity” of the instantiation is discarded
   2. Generate a sample calling the LLM with these inputs
   3. The prompt enforces four simultaneous constraints:
      1. The sample must align with instantiation → diversity
      2. The label must align with the expected label → control
      3. The sample must match the example's domain and style → realism
      4. The sample must be a BOUNDARY CASE, not a trivial one.  
        "stress-test a smart and successful classifier".  
          - Trick: the generator also emits the reasoning = the justification of the label.  
          - Trick: no meta-leakage allowed ("do not mention test cases, models, dimensions, or labels in your output")  
          - Note: If the generated sample label is not the target one, the divergence is logged and ignored.  
   4. Debate label
      - Two judges take the sample (text and label). One judge is precision-oriented, the other is recall-oriented. A judge provides a verdict (reasoning, label, confidence).
      - Round 1:
        - each judge classifies the text → get a judge label
        - Sample label = judge1 label = judge2 label → accept sample
      - Round 2 if not consensus:
        - Each judge receives its previous verdict + verdict of the other judge + reasoning of the sample (= why this sample should be like that)
        - Consensus? → accept sample or return the dissenting feedbacks
   5. Refine sample
     At this step, the sample was rejected and the judges provided feedback.  
     A new sample is generated, and this sample is debated.  
     = a sample is never accepted without a debate.  
4. Profit!

> [!NOTE]
> What is **Verbalized Sampling**?
> 
> Do not ask for a list, ask for a “distribution”.  
> = instead of a list of strings (the instantiations), ask the LLM for a description of the instantiation, a polarity (true/false/both) and a score (probability). We don’t use the polarity, nor the score afterward.  
> 
> Unlock **Diversity** and pushes beyond typical modes.
> 
> (This technique is at the core of BARRED and is awesome!)  
> 
>Link to the [Paper "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"](https://arxiv.org/abs/2510.01171)

## Paper fidelity, and its gray zone

As an independent reimplementation, I tried to keep the paper fidelity as much as possible.

I audited the code against the paper, line by line. The four steps are all there, and the prompts follow the Appendix
almost word for word.

### Where I diverge

- **The generated label wins over the drawn one.**  
  Algorithm 1 draws a target label `y`, then debates and refines against that `y`.
  But the generation prompt (A.2) asks the LLM for "the label you believe it should get", so the LLM can flip it.
  Here, the flip is logged, and the generated label becomes the authority for the debate and the refinement.
  Consequence: the 50/50 balance of the dataset is not guaranteed.  
  (The tension lives in the paper itself)
- **Three guidelines dropped from the dimension extraction prompt** (A.1):
  I think those guidelines were task-specific, introduced by the authors for the tasks they were working on.
  So "position in the input block", "computations / number of occurrences" and "jailbreak scenarios for transcripts" were removed.
- **Temperatures and seeds are mine** (the paper says nothing): `temperature=1.0` and `seed=None` to generate and refine (diversity), `temperature=0.0` and `seed=0` to judge and deduplicate (stability).

### The gray zones

Places where the paper is silent, and I had to pick an interpretation:

- **The Advocate never calls the LLM.** The paper describes an advocate defending the sample, "rigid, never changing
  position". So either the Advocate is a LLM call or just "fields" added to the prompt to the judges.
- **`R_max` is not given** (`T = 2` debate rounds is). Default here: two refinement rounds.
- **`{target_dimension}` in the generation prompt**: the dimension, the sampled instantiation, or both?
  I pass the instantiation description only.
- **"Filter out semantically similar dimensions"**, method not described: I use an LLM pass, replaying the
  conversation. Maybe the authors loop on the seed examples, generated multiple lists of dimensions, then deduplicating those lists into one. In the code, I pass all examples, no a chunk of them, nor some of them randomly.

## The paper, its authors, and resources

Paper authors:
- [Arnon Mazza](https://www.linkedin.com/in/arnon-mazza-4471424/)
- [Elad Levi](https://www.linkedin.com/in/elad-levi-a938a3121/)

Company: [Plurai.ai](https://www.plurai.ai)

The main article giving a high-level overview of BARRED (April 28, 2026): [Introducing BARRED: turn any policy prompt into a high-accuracy efficient guardrail](https://www.plurai.ai/blog/introducing-barred-turn-any-policy-prompt-into-a-high-accuracy-efficient-guardrail)

> [!NOTE]
> I highly recommend giving Plurai.ai a shot to create a classifier. 
> Under the hood, the app uses (I think) a derived version of BARRED. 
> The UI and the experience are great.  
> [Try it out at plurai.ai](https://www.plurai.ai).

### Citation

The paper this library implements:

```bibtex
@misc{mazza2026barred,
  title         = {BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate},
  author        = {Arnon Mazza and Elad Levi},
  year          = {2026},
  eprint        = {2604.25203},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.25203}
}
```

## Bonus: Make your own BARRED logo

[![Logo Generator](https://raw.githubusercontent.com/tomsquest/barred/main/doc/logo_generator.png)](https://raw.githack.com/tomsquest/barred/main/doc/logo_generator.html)

## Changelog

Changelog and releases are on [GitHub Releases](https://github.com/tomsquest/barred/releases).

## Development

### Setup

Install after cloning the repository:

```bash
just install
```

Check everything (lint, type, tests...):

```bash
just checks
```

### Release

Releases are triggered by a version tag: pushing `v*` runs `.github/workflows/release.yml`,
which builds, smoke-tests the wheel and the sdist, then publishes to PyPI via Trusted
Publishing (no token involved).

`just release` recipe does the whole sequence — bump, check, commit, tag, push:

```bash
just release minor   # 0.2.0 => 0.3.0
just release patch   # 0.2.0 => 0.2.1
just release rc      # 0.2.0 => 0.2.0rc1, published as a pre-release
just release 1.0.0   # explicit version
```

It refuses to start from a dirty tree or outside `main`, and rolls the bump back if
anything fails, so a failed run leaves nothing behind.
