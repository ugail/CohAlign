"""Verify every headline number in the CohAlign paper from the precomputed
tables in Results/.

Usage:  python verify_results.py
Needs only NumPy and pandas (requirements-verify.txt), runs in seconds,
and prints a PASS/FAIL line for every claim.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "Results"
checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- Phase FD
fd = pd.read_csv(RES / "phaseFD_firstorder.csv")
diss = fd[fd["case"] == "dissipative"]
check("FD dissipative suite: max relative error 3.4e-3 (probe bias)",
      abs(diss["rel_err"].max() - 3.415903e-03) < 1e-6,
      f"max rel err {diss['rel_err'].max():.3e}")
check("FD amplitude damping essentially exact",
      diss[diss.channel == "amp_damp"]["rel_err"].iloc[0] < 1e-7)
prot = fd[fd["case"] == "protected"]
check("Protected control: |Lambda_FD| below 1e-10, relative error NaN",
      abs(prot["lambda_fd_extrap"].iloc[0]) < 1e-10
      and np.isnan(prot["rel_err"].iloc[0]))
sg = fd[fd["case"] == "signed"].iloc[0]
check("Signed case: Lambda_resp -0.5837 vs FD -0.5837 on 400 matched draws",
      round(sg["lambda_resp"], 4) == -0.5837
      and round(sg["lambda_fd_extrap"], 4) == -0.5837
      and int(sg["n_draws"]) == 400,
      f"rel err {sg['rel_err']:.2e}")
check("Signed bootstrap interval excludes zero",
      sg["lambda_resp_ci_hi"] < 0.0,
      f"({sg['lambda_resp_ci_lo']:.4f}, {sg['lambda_resp_ci_hi']:.4f})")

# ---------------------------------------------------------------- Phase K
k = pd.read_csv(RES / "phaseK_param_map.csv")
kf = k[np.isfinite(k["rel_err"])]
check("Parameter map: fourteen measurable parameter and channel pairs",
      len(kf) == 14, f"{len(kf)} finite of {len(k)} recorded")
check("Parameter map: every measurable point at the probe bias",
      kf["rel_err"].max() < 1.1e-3, f"max {kf['rel_err'].max():.2e}")
spread = kf["lambda_resp"].max() / kf["lambda_resp"].min()
check("Parameter map: susceptibilities span more than a factor of three",
      spread > 3.0, f"{kf['lambda_resp'].min():.2f} to "
      f"{kf['lambda_resp'].max():.2f}")
dep = kf[kf.channel == "dephase"].set_index(["ell", "q"])["lambda_resp"]
cor = kf[kf.channel == "corr_dephase"].set_index(["ell", "q"])["lambda_resp"]
shared = dep.index.intersection(cor.index)
dep_max = set(dep[shared][np.isclose(dep[shared], dep[shared].max())].index)
cor_min = set(cor[shared][np.isclose(cor[shared], cor[shared].min())].index)
check("Parameter map: dephasing-fastest locations are control-most-protected",
      dep_max <= cor_min,
      f"{sorted(dep_max)} within {sorted(cor_min)}")

# ---------------------------------------------------------------- Phase C
z = pd.read_csv(RES / "phaseC_zero_alignment.csv")
check("Zero-alignment control: operator mode ratio 0.969 at one parameter",
      abs(z["omega_op"].max() - 0.9688) < 5e-4,
      f"{z['omega_op'].max():.4f}")
check("Zero-alignment control: every parameter predicted protected with "
      "measured degradation below 1e-13",
      bool(z["protected_pred"].all())
      and z["delta_meas"].abs().max() < 1e-13)

# ---------------------------------------------------------------- Phase F
f = pd.read_csv(RES / "phaseF_validation.csv")
per = f[f.channel != "coh_diss_mix"].groupby("channel").agg(
    pred=("omega_pred", "first"), hat=("omega_hat_extrap", "first"),
    sm=("omega_hat_small", "first"))
rmse = float(np.sqrt(np.mean((per["hat"] - per["pred"]) ** 2)))
check("Family RMSE 0.009 over the eight distinct channels",
      abs(rmse - 0.00875) < 2e-4, f"{rmse:.5f}")
raw = float(np.sqrt(np.mean((per["sm"] - per["pred"]) ** 2)))
check("Raw finite-noise extraction contrast 0.075", abs(raw - 0.0748) < 2e-3)
null = f[f.channel == "coh_diss_mix"].groupby("channel")["omega_pred"].first()
amp = f[f.channel == "amp_damp"].groupby("channel")["omega_pred"].first()
check("Null coherent control audits identically to amplitude damping",
      abs(null.iloc[0] - amp.iloc[0]) < 1e-9)

# ---------------------------------------------------------------- Phase G
g = pd.read_csv(RES / "phaseG_regression.csv")
resp = g[g.model == "response_rate"].sort_values("window_defect",
                                                 ascending=False)
check("Regression: response model R2 0.993 / 0.998 / 0.999 as the window "
      "tightens",
      np.allclose(resp["R2"].values, [0.9926, 0.9978, 0.9994], atol=5e-4))
check("Regression: response RMSE falls 0.094 to 0.023",
      abs(resp["RMSE"].iloc[0] - 0.0937) < 2e-3
      and abs(resp["RMSE"].iloc[-1] - 0.0227) < 2e-3)
loco = pd.read_csv(RES / "phaseG_loco.csv")
check("LOCO: pooled 0.057, worst channel 0.083",
      abs(float((loco["loco_rmse"] ** 2).mean() ** 0.5) - 0.057) < 2e-3
      and abs(loco["loco_rmse"].max() - 0.083) < 2e-3)

# ---------------------------------------------------------------- Phase I
i2 = pd.read_csv(RES / "phaseI_r2_sector.csv")
x = i2[i2.channel != "coh_diss_mix"]
r2rmse = float(np.sqrt(np.mean((x["omega_hat"] - x["omega_pred"]).dropna() ** 2)))
check("Two-excitation sector RMSE 0.037 over eight distinct channels",
      abs(r2rmse - 0.0371) < 5e-4, f"{r2rmse:.4f}")

# ---------------------------------------------------------------- Phase B
b = pd.read_csv(RES / "phaseB_probe_convergence.csv")
dseq = b[b.channel == "dephase"].sort_values("gamma_probe", ascending=False)
check("Probe convergence: dephasing 3.960, 3.9960, 3.99960, 3.99996",
      np.allclose(dseq["lambda_coh"].values,
                  [3.960, 3.9960, 3.99960, 3.99996], atol=1e-4))

# ---------------------------------------------------------------- manifest
mf = json.load(open(RES / "paper_run_manifest.json"))
check("Paper-run manifest present with module, CSV and figure hashes",
      mf.get("mode") == "paper"
      and len(mf.get("module_sha256_16", {})) == 3
      and len(mf.get("output_csv_sha256_16", {})) >= 20
      and len(mf.get("figure_sha256_16", {})) >= 8)

print("=" * 72)
n_pass = sum(checks)
print(f"{n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
