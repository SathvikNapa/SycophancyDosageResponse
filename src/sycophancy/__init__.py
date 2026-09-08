"""Sycophancy dosage-response experiment library.

Pickle backward-compatibility shim
-----------------------------------
Before this repo was reorganized into a ``src/sycophancy`` package, these
modules lived at the repo root and were imported as bare top-level modules
(``import config``, ``import reasoning_uncertainty``, etc.). ``pickle``
embeds a class's *original* module path at dump time, so every historical
``experiment_out/**/*.pkl`` that contains an instance of a class defined in
one of those modules (most notably ``ReasoningTrace``/``ReasoningStep`` in
``reasoning_uncertainty.py``) can only be unpickled if that bare module
name is still resolvable.

Registering these aliases in ``sys.modules`` (once, here, at package import
time) keeps every previously-generated pickle loadable without needing to
regenerate any experiment data. ``setdefault`` is used so this never
clobbers an unrelated same-named module that happened to be imported
first. Do **not** remove this shim without re-running the pickle audit in
``REPRODUCIBILITY.md``.
"""

import sys as _sys

from . import build_analysis_dfs as _build_analysis_dfs
from . import calibrator as _calibrator
from . import config as _config
from . import data as _data
from . import entropy as _entropy
from . import generator as _generator
from . import reasoning_uncertainty as _reasoning_uncertainty
from . import sycophancy_dosage as _sycophancy_dosage

_LEGACY_FLAT_MODULE_ALIASES = {
    "config": _config,
    "data": _data,
    "generator": _generator,
    "entropy": _entropy,
    "calibrator": _calibrator,
    "reasoning_uncertainty": _reasoning_uncertainty,
    "sycophancy_dosage": _sycophancy_dosage,
    "build_analysis_dfs": _build_analysis_dfs,
}
for _name, _module in _LEGACY_FLAT_MODULE_ALIASES.items():
    _sys.modules.setdefault(_name, _module)

del _sys, _name, _module, _LEGACY_FLAT_MODULE_ALIASES
