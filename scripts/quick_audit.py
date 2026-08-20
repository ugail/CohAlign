"""Minimal worked example: audit one parameter of a built-in channel, then
drop in a custom channel of your own.

Run from the repository root:  python scripts/quick_audit.py
Takes a few seconds on a laptop (six qubits, three layers).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cohalign import (  # noqa: E402
    Brickwork, audit_parameter, build_channel_suite,
    prepare_channel_context, teacher_background,
)
from cohalign.cohalign_core import NoiseChannel  # noqa: E402

n, L, r = 6, 3, 1
model = Brickwork(n, L, r)
beta = teacher_background(42, L, n)

# ---- audit a built-in channel at one parameter -------------------------
channel = build_channel_suite(n)["dephase"]
ctx = prepare_channel_context(channel, n, r, gamma_probe=1e-3)
report = audit_parameter(ctx, model, beta, ell_i=1, q_i=0,
                         n_theta=20, seed=1)
print("dephasing at parameter (1, 0)")
print(f"  worst-case rate      lambda_coh  = {report['lambda_coh_worst']:.4f}")
print(f"  mode contraction     lambda_mode = {report['lambda_mode']:.4f}")
print(f"  response             Lambda_resp = {report['lambda_response']:.4f}")
print(f"  bootstrap interval               = "
      f"({report['lambda_response_ci'][0]:.4f}, "
      f"{report['lambda_response_ci'][1]:.4f})")
print(f"  status: {report['normalisation_status']}   "
      f"signed-only: {report['signed_susceptibility_only']}")

# ---- drop in a custom channel ------------------------------------------
class MyDampedDephasing(NoiseChannel):
    """Example custom channel: dephasing with a site-dependent strength.
    Correlated channels override apply() directly, exactly like this,
    rather than supplying single-qubit Kraus operators."""
    name = "my_damped_dephasing"

    def apply(self, M, gamma, n):
        from cohalign.cohalign_core import apply_single_qubit_kraus
        out = M
        for q in range(n):
            g = gamma * (0.5 + q / (n - 1))
            K0 = np.sqrt(1 - g) * np.eye(2)
            K1 = np.sqrt(g) * np.diag([1.0, -1.0])
            out = apply_single_qubit_kraus(out, [K0, K1], q, n)
        return out


ctx2 = prepare_channel_context(MyDampedDephasing(), n, r, gamma_probe=1e-3)
report2 = audit_parameter(ctx2, model, beta, ell_i=1, q_i=0,
                          n_theta=20, seed=1)
print("\ncustom channel at the same parameter")
print(f"  Lambda_resp = {report2['lambda_response']:.4f}   "
      f"status: {report2['normalisation_status']}")
