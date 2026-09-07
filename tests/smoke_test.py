import importlib

import barred

# Every name the package promises is actually reachable.
missing = [name for name in barred.__all__ if not hasattr(barred, name)]
assert not missing, f"listed in __all__ but not importable: {missing}"

# Every module is shipped in the wheel, not merely present in the source tree.
for module in [
    "barred.debate",
    "barred.decompose_dimensions",
    "barred.exceptions",
    "barred.generate_sample",
    "barred.llm",
    "barred.logger",
    "barred.observer",
    "barred.pipeline",
    "barred.refine_sample",
    "barred.types",
]:
    importlib.import_module(module)

# The public entry points survived the round trip as callables.
for entry_point in ["barred", "decompose_dimensions", "LLM"]:
    assert callable(getattr(barred, entry_point)), f"{entry_point} is not callable"
