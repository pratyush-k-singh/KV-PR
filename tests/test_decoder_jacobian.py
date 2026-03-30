"""Unit tests for the P20 decoder-Jacobian power-iteration estimator.

Validation strategy: power_iterate_top_sigma takes a callable run_fn whose
Jacobian-at-origin is the operator we are estimating. We pin run_fn to
synthetic linear and weakly-nonlinear maps with known top singular value,
then assert the estimate converges within tolerance. This validates the
math (finite-difference JVP + autograd VJP + power iteration) without
needing a full transformer in the loop.
"""

from __future__ import annotations

import unittest

import torch

from src.dart_pagedkv.power_iteration import power_iterate_top_sigma


def _make_linear_run_fn(A: torch.Tensor):
    """Returns run_fn(p) = A @ p (Jacobian-at-origin is A)."""

    def run_fn(p: torch.Tensor) -> torch.Tensor:
        return A @ p

    return run_fn


def _top_sigma_via_svd(A: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(A)[0].item())


class TestPowerIterationLinearOperator(unittest.TestCase):
    """Linear-map sanity: estimator must recover sigma_max(A) within tol."""

    def setUp(self):
        torch.manual_seed(2026)

    def test_top_sigma_diagonal_operator(self):
        # Diagonal A with descending eigenvalues -> top sigma = max diag entry.
        diag = torch.tensor([5.0, 3.0, 1.5, 0.7, 0.1], dtype=torch.float32)
        A = torch.diag(diag)
        sigma_true = float(diag[0].item())
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=A.shape[1], device="cpu",
            n_iter=20, eps=1e-3,
        )
        self.assertAlmostEqual(result["top_sigma"], sigma_true, delta=0.02)

    def test_top_sigma_random_square_operator(self):
        # Random 16x16 operator; top sigma must match SVD result.
        d = 16
        A = torch.randn(d, d, dtype=torch.float32)
        sigma_true = _top_sigma_via_svd(A)
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=30, eps=1e-3,
        )
        rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
        self.assertLess(rel_err, 0.01)

    def test_top_sigma_rectangular_operator(self):
        # Rectangular operator (input space larger than output).
        out_dim, in_dim = 8, 32
        A = torch.randn(out_dim, in_dim, dtype=torch.float32)
        sigma_true = _top_sigma_via_svd(A)
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=in_dim, device="cpu", n_iter=30, eps=1e-3,
        )
        rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
        self.assertLess(rel_err, 0.01)

    def test_top_sigma_low_rank_operator(self):
        # Rank-1 operator: A = u v^T with ||u||=2, ||v||=3 -> sigma_top=6.
        u = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32) * (2.0 / 2.0**0.5)
        v = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32) * (3.0 / 3.0**0.5)
        A = u.unsqueeze(1) @ v.unsqueeze(0)
        sigma_true = _top_sigma_via_svd(A)  # ~ 6.0
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=4, device="cpu", n_iter=30, eps=1e-3,
        )
        rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
        self.assertLess(rel_err, 0.02)

    def test_top_sigma_zero_operator(self):
        # Identically-zero operator: estimator should report ~0, not crash.
        d = 8
        A = torch.zeros(d, d, dtype=torch.float32)
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=10, eps=1e-3,
        )
        self.assertLess(result["top_sigma"], 1e-6)


class TestPowerIterationContractiveVsExpansive(unittest.TestCase):
    """The estimator's qualitative outputs (sigma < 1 vs >= 1) must match
    the paper's contractive-vs-expansive regime classification."""

    def setUp(self):
        torch.manual_seed(2027)

    def test_contractive_regime_detection(self):
        # sigma_top = 0.7 (contractive) — model the Phi-snapkv regime.
        d = 12
        A = 0.7 * torch.eye(d, dtype=torch.float32)
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=15, eps=1e-3,
        )
        self.assertLess(result["top_sigma"], 1.0)
        self.assertAlmostEqual(result["top_sigma"], 0.7, delta=0.02)

    def test_expansive_regime_detection(self):
        # sigma_top = 1.4 (expansive) — model the LLaMA-snapkv regime.
        d = 12
        A = 1.4 * torch.eye(d, dtype=torch.float32)
        run_fn = _make_linear_run_fn(A)
        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=15, eps=1e-3,
        )
        self.assertGreaterEqual(result["top_sigma"], 1.0)
        self.assertAlmostEqual(result["top_sigma"], 1.4, delta=0.02)


class TestPowerIterationConvergence(unittest.TestCase):
    """Convergence behavior: top_sigma estimate stabilizes within n_iter."""

    def setUp(self):
        torch.manual_seed(2028)

    def test_convergence_with_spectral_gap(self):
        # Spectral gap of 5x -> rapid convergence (~few iterations).
        d = 8
        diag = torch.tensor([5.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])
        A = torch.diag(diag)
        run_fn = _make_linear_run_fn(A)
        res5 = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=5, eps=1e-3,
        )
        res20 = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=20, eps=1e-3,
        )
        # Both should match the SVD top sigma within 1% with this spectral gap.
        sigma_true = _top_sigma_via_svd(A)
        self.assertLess(abs(res5["top_sigma"] - sigma_true) / sigma_true, 0.01)
        self.assertLess(abs(res20["top_sigma"] - sigma_true) / sigma_true, 0.01)

    def test_eps_sensitivity_in_central_difference(self):
        # Central difference is exact for linear operators in real arithmetic;
        # float32 cancellation tightens the usable eps range. The producer's
        # default eps=1e-3 is the sweet spot. Test eps in [1e-4, 1e-2]: looser
        # tolerance at the eps=1e-4 end where cancellation noise dominates.
        d = 8
        A = torch.randn(d, d, dtype=torch.float32)
        sigma_true = _top_sigma_via_svd(A)
        run_fn = _make_linear_run_fn(A)
        tolerances = {1e-2: 0.02, 1e-3: 0.02, 1e-4: 0.05}
        for eps, tol in tolerances.items():
            result = power_iterate_top_sigma(
                run_fn, total_dim=d, device="cpu", n_iter=20, eps=eps,
            )
            rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
            self.assertLess(
                rel_err, tol,
                msg=f"eps={eps}: rel_err={rel_err:.4f} (tolerance {tol})",
            )


class TestPowerIterationProbeScale(unittest.TestCase):
    """The per_entry_scale knob is the bf16 fix. For linear operators the
    two parametrisations must produce the same top sigma (FD is exact
    regardless of probe magnitude); per_entry_scale=True is the bf16-safe
    default for the producer."""

    def setUp(self):
        torch.manual_seed(2030)

    def test_per_entry_scale_equivalence_on_linear_op(self):
        d = 64
        A = torch.randn(d, d, dtype=torch.float32)
        sigma_true = float(torch.linalg.svdvals(A)[0].item())
        run_fn = _make_linear_run_fn(A)
        # Both parametrisations should recover sigma_true; their internal
        # probe magnitudes differ by a sqrt(d) factor but the FD ratio
        # cancels it for linear operators.
        a = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu",
            n_iter=20, eps=0.05, per_entry_scale=True,
        )
        b = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu",
            n_iter=20, eps=0.05, per_entry_scale=False,
        )
        # Each individually correct (linear FD is exact)
        self.assertLess(abs(a["top_sigma"] - sigma_true) / sigma_true, 0.02)
        self.assertLess(abs(b["top_sigma"] - sigma_true) / sigma_true, 0.02)
        # And consistent with each other
        self.assertLess(abs(a["top_sigma"] - b["top_sigma"]) / sigma_true, 0.02)
        # Reported probe_norm differs by sqrt(d)
        self.assertAlmostEqual(a["probe_norm"], b["probe_norm"] * (d ** 0.5), places=4)


class TestPowerIterationP20bRobustness(unittest.TestCase):
    """P20b: unit-norm grad and NaN recovery. Regression coverage for the
    LLaMA NaN failure mode in the 2026-05-22 P20 run."""

    def setUp(self):
        torch.manual_seed(2032)

    def test_unit_norm_grad_matches_full_grad_on_linear_op(self):
        # For linear operators, J^T is linear in its argument, so scaling
        # the grad_outputs by ||jv|| then dividing the result by ||jv||
        # should be a no-op. Both modes must recover sigma_top equally.
        d = 32
        A = torch.randn(d, d, dtype=torch.float32) * 5.0  # larger scale
        sigma_true = float(torch.linalg.svdvals(A)[0].item())
        run_fn = _make_linear_run_fn(A)
        a = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=20, eps=0.05,
            unit_norm_grad=True,
        )
        b = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=20, eps=0.05,
            unit_norm_grad=False,
        )
        self.assertLess(abs(a["top_sigma"] - sigma_true) / sigma_true, 0.02)
        self.assertLess(abs(b["top_sigma"] - sigma_true) / sigma_true, 0.02)

    def test_nan_in_jvp_recovers_via_resample(self):
        # Make a run_fn that returns NaN on the first call and a clean
        # linear map afterwards. Power iteration should detect the NaN,
        # resample v, and continue. Final top_sigma should be a real
        # number (not NaN), and n_resamples should reflect the detection.
        d = 16
        A = torch.randn(d, d, dtype=torch.float32)
        sigma_true = float(torch.linalg.svdvals(A)[0].item())
        call_count = {"n": 0}

        def run_fn(p):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return torch.full((d,), float("nan"))
            return A @ p

        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=10, eps=0.05,
            detect_nan=True,
        )
        import math as _math
        # First JV is NaN, so n_resamples >= 1
        self.assertGreaterEqual(result["n_resamples"], 1)
        # Subsequent iterations should recover; final top_sigma is finite
        self.assertFalse(_math.isnan(result["top_sigma"]))
        rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
        self.assertLess(rel_err, 0.1)


class TestPowerIterationNonLinear(unittest.TestCase):
    """Non-linear maps: the estimator must recover the Jacobian-at-origin
    top sigma, not be confused by higher-order terms."""

    def setUp(self):
        torch.manual_seed(2029)

    def test_quadratic_nonlinearity_around_zero(self):
        # run_fn(p) = A @ p + 0.5 * (p^T p) * c
        # Jacobian at zero is A (the quadratic term has zero gradient at 0).
        d = 8
        A = torch.randn(d, d, dtype=torch.float32)
        c = torch.randn(d, dtype=torch.float32)
        sigma_true = _top_sigma_via_svd(A)

        def run_fn(p):
            return A @ p + 0.5 * (p @ p) * c

        result = power_iterate_top_sigma(
            run_fn, total_dim=d, device="cpu", n_iter=20, eps=1e-3,
        )
        rel_err = abs(result["top_sigma"] - sigma_true) / sigma_true
        # Quadratic term contributes O(eps^2) at +-eps; at eps=1e-3 the
        # contribution is small relative to a typical sigma ~ O(d^0.5).
        self.assertLess(rel_err, 0.05)


if __name__ == "__main__":
    unittest.main()
