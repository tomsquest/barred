<p align="center">
<img src="https://raw.githubusercontent.com/tomsquest/barred/main/doc/cover.png" alt="Cover" />
</p>

# BARRED: generate faithful and diverse training sets

[![PyPI](https://img.shields.io/pypi/v/barred?color=blue)](https://pypi.org/project/barred/)
[![Python versions](https://img.shields.io/pypi/pyversions/barred)](https://pypi.org/project/barred/)
[![License](https://img.shields.io/github/license/tomsquest/barred?color=green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.25203-b31b1b)](https://arxiv.org/abs/2604.25203)

**Unofficial** implementation of the BARRED paper:

> Boundary Alignment Refinement through REflection and Debate, aka BARRED
> [arXiv:2604.25203](https://arxiv.org/pdf/2604.25203)

> [!NOTE]
> I developed this library for my own needs without endorsement. 
> It is not affiliated with the authors of the paper.

## What is BARRED?

**In a nutshell**: BARRED is a framework for generating **faithful** and **diverse** synthetic training data using only
a task description and a small set of unlabeled examples.

### Step by step

1. Define the task and give examples to know what to generate Criterion:
    ```
    Criterion: `True means this sentence is positive, False otherwise`
    Examples: `I love it`, `it's rainy`, `My uncle bobby is a good boy`
    ```
2. Let the library decompound the problem into dimensions
3. Then the library generates the training set
4. Results: a set of *labeled* sentences
    ```
       1. `My mother is a wonderful person -> true`
       2. `Her sister is boring -> false`
    ```

### Show me some code

```python
#
# Step 1. Task definition
#
criterion = "True when the sentence expresses a positive sentiment, False otherwise"
examples = [
   "The delivery arrived two days late and the box was crushed.",
   "Honestly one of the best purchases I've made this year.",
   "It works, I guess.",
]

#
# Step 2. Decompose the Criterion into Dimensions
#
dimensions = await decompose_dimensions(llm, criterion=criterion, examples=examples)

#
# Step 3. Generate Samples
#
samples = [
   sample
   async for sample in barred(
      llm,
      criterion=criterion,
      examples=examples,
      dimensions=dimensions,
      num_samples=5,
   )
]
```


## Notebook/Demo

Open the Demo notebook in [Google Colab](https://colab.research.google.com/github/tomsquest/barred/blob/main/notebooks/demo_sentiment_analysis.ipynb).

## Why it's cool?

TODO

## Installation

```
uv add barred

# with Pip
pip install barred
```

## Changelog/Releases

Changelog and releases are on [GitHub Releases](https://github.com/tomsquest/barred/releases).

## Development setup

Install after cloning the repository:

```bash
just install
```

Check everything (lint, type, tests...):

```bash
just checks
```

## Release

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
