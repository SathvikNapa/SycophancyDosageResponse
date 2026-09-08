"""
Smoke test: every sycophancy submodule imports cleanly, and the legacy flat
module-name aliases (needed to unpickle pre-reorg experiment_out/*.pkl
files — see src/sycophancy/__init__.py) actually resolve.
"""

import importlib
import pkgutil
import sys

import sycophancy


def test_every_submodule_imports():
    failures = []
    for m in pkgutil.iter_modules(sycophancy.__path__):
        try:
            importlib.import_module(f"sycophancy.{m.name}")
        except Exception as e:  # pragma: no cover - failure path only
            failures.append((m.name, repr(e)))
    assert not failures, failures


def test_legacy_flat_module_aliases_resolve():
    import build_analysis_dfs
    import calibrator
    import config
    import data
    import entropy
    import generator
    import reasoning_uncertainty
    import sycophancy_dosage

    assert reasoning_uncertainty is sys.modules["sycophancy.reasoning_uncertainty"]
    assert config is sys.modules["sycophancy.config"]
    assert data is sys.modules["sycophancy.data"]
    assert generator is sys.modules["sycophancy.generator"]
    assert entropy is sys.modules["sycophancy.entropy"]
    assert calibrator is sys.modules["sycophancy.calibrator"]
    assert sycophancy_dosage is sys.modules["sycophancy.sycophancy_dosage"]
    assert build_analysis_dfs is sys.modules["sycophancy.build_analysis_dfs"]
