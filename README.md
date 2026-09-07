<p align="center">
<img src="https://raw.githubusercontent.com/tomsquest/barred/main/doc/cover.png" alt="Cover" />
</p>

# BARRED: generate faithful and diverse training sets

Unofficial implementation of the BARRED paper:

> Boundary Alignment Refinement through REflection and Debate, aka BARRED
> [arXiv:2604.25203](https://arxiv.org/pdf/2604.25203)

## What is BARRED?

**In a nutshell**: BARRED is a framework for generating **faithful** and **diverse** synthetic training data using only
a task description and a small set of unlabeled examples.

**Step by step**:

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

## Why it's cool?

TODO

## Installation

```
uv add barred

# with Pip
pip install barred
```

## Usage

TODO

## Development setup

Install after cloning the repository:

```bash
just install
```

Check everything (lint, type, tests...):

```bash
just checks
```
