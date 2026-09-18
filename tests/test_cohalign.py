"""Structural invariant tests for CohAlign.

These run in a couple of minutes on a laptop at small system sizes and
verify the estimator identities, the control constructions and the public
interface, independently of the paper's precomputed tables.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cohalign import (  # noqa: E402
    Brickwork, audit_parameter, build_channel_suite,
    prepare_channel_context, teacher_background,
)
from cohalign import bench as cab  # noqa: E402
from cohalign import rates as car  # noqa: E402

N, L, R = 6, 3, 1
MODEL = Brickwork(N, L, R)
BETA = teacher_background(42, L, N)
SUITE = build_channel_suite(N)


def _cost(theta):
    return MODEL.output(MODEL.evolve_dm(theta, BETA))


def test_trace_preservation():
    rng = np.random.default_rng(0)
    dim = 2 ** N
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    for name, ch in SUITE.items():
        out = ch.apply(rho, 0.05, N)
        assert abs(np.trace(out) - 1.0) < 1e-12, name


def test_parameter_shift_matches_direct_derivative():
    rng = np.random.default_rng(3)
    theta = 0.05 * rng.normal(size=(L, N))
    ell_i, q_i = 1, 0
    tp, tm = theta.copy(), theta.copy()
    tp[ell_i, q_i] += np.pi / 2
    tm[ell_i, q_i] -= np.pi / 2
    g_shift = 0.5 * (_cost(tp) - _cost(tm))
    eps = 1e-6
    tp, tm = theta.copy(), theta.copy()
    tp[ell_i, q_i] += eps
    tm[ell_i, q_i] -= eps
    g_fd = (_cost(tp) - _cost(tm)) / (2 * eps)
    assert abs(g_shift - g_fd) < 1e-8


def test_null_coherent_control_matches_amplitude_damping():
    vals = {}
    for name in ("amp_damp", "coh_diss_mix"):
        ctx = prepare_channel_context(SUITE[name], N, R, 1e-3)
        rep = audit_parameter(ctx, MODEL, BETA, 1, 0, n_theta=6, seed=1)
        vals[name] = rep["lambda_response"]
    assert abs(vals["amp_damp"] - vals["coh_diss_mix"]) < 1e-9


def test_matched_finite_difference_identity():
    ctx = prepare_channel_context(SUITE["dephase"], N, R, 1e-3)
    rep = audit_parameter(ctx, MODEL, BETA, 1, 0, n_theta=8, seed=1)
    lam = rep["lambda_response"]
    h = 1e-4
    m = cab.paired_degradation(MODEL, BETA, 1, 0, SUITE["dephase"], h,
                               n_theta=8, delta_init=0.05, seed=1,
                               shift=np.pi / 2, bootstrap_B=20, boot_seed=7)
    lam_fd = m["delta_tilde"] / (2 * h)
    assert abs(lam_fd - lam) / abs(lam) < 5e-3  # probe bias plus O(h)


def test_inactive_parameter_reports_cleanly():
    ctx = prepare_channel_context(SUITE["dephase"], N, R, 1e-3)
    theta_draws = np.zeros((5, L, N))  # zero phases give zero gradients
    rep = audit_parameter(ctx, MODEL, BETA, 1, 0, n_theta=5, seed=1,
                          theta_draws=theta_draws)
    assert np.isnan(rep["lambda_response"])
    assert "bootstrap_B" in rep and "bootstrap_seed" in rep


def test_context_reuse_does_not_rebuild_generator():
    ctx = prepare_channel_context(SUITE["dephase"], N, R, 1e-3)
    gen_before = ctx["L_r"]
    audit_parameter(ctx, MODEL, BETA, 1, 0, n_theta=4, seed=1)
    audit_parameter(ctx, MODEL, BETA, 1, 1, n_theta=4, seed=1)
    assert ctx["L_r"] is gen_before


def test_deprecated_alias_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        car.lambda_vis_op(MODEL, BETA, SUITE["dephase"], 1, 0,
                          n_theta=3, seed=1, gamma_probe=1e-3)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_signed_channel_flagged_without_ratio():
    from cohalign.cohalign_core import SiteZOverRotation
    ctx = prepare_channel_context(SiteZOverRotation(), N, R, 1e-3)
    rep = audit_parameter(ctx, MODEL, BETA, 1, 0, n_theta=6, seed=1)
    assert rep["signed_susceptibility_only"]
    om = rep.get("omega_response")
    assert om is None or (isinstance(om, float) and np.isnan(om))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_chain_topology_drops_cycle_edge():
    ring = Brickwork(N, L, R, topology="ring")
    chain = Brickwork(N, L, R, topology="chain")
    for ell in range(L):
        assert set(chain.E[ell]) == {j for j in ring.E[ell] if j < N - 1}
    assert any(N - 1 in ring.E[ell] for ell in range(L))


def test_readout_site_moves_the_readout():
    m0 = Brickwork(N, L, R, readout_site=0)
    m2 = Brickwork(N, L, R, readout_site=2)
    assert not np.allclose(m0.readout, m2.readout)
    assert np.allclose(m2.readout, m2.readout.conj().T)
