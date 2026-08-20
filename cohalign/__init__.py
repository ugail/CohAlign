"""CohAlign: first-order noise susceptibility of the ensemble gradient
second moment in equivariant quantum neural networks.

The three modules are materialised verbatim from the pipeline notebook and
use flat imports, so this package prepends its own directory to sys.path
and re-exports the public API.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import cohalign_core as core          # noqa: E402
import cohalign_rates as rates        # noqa: E402
import cohalign_bench as bench        # noqa: E402

from cohalign_rates import (          # noqa: E402,F401
    prepare_channel_context, audit_parameter, lambda_mode, lambda_vis_resp,
)
from cohalign_core import build_channel_suite, Brickwork, teacher_background  # noqa: E402,F401

__version__ = "1.8.1"
