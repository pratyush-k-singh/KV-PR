"""Power-iteration top singular value estimator (used by P20 / decoder_jacobian).

Extracted into a standalone module so the math primitive can be unit-tested
without dragging in the producer / model / data-loader import chain.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def power_iterate_top_sigma(
    run_fn,
    *,
    total_dim: int,
    device: torch.device | str,
    n_iter: int = 10,
    eps: float = 0.05,
    eps_floor: float = 1e-12,
    rng: Optional[torch.Generator] = None,
    per_entry_scale: bool = True,
    unit_norm_grad: bool = True,
    detect_nan: bool = True,
) -> dict:
    """Power-iterate the top singular value of the linearization of ``run_fn``
    at the origin via central-difference JVPs and autograd VJPs.

    ``run_fn(perturbation: Tensor[total_dim]) -> Tensor[output_dim]`` must be
    differentiable in its input on its device of choice; it is called with
    perturbations of the form ``+/- p`` for various ``p``.

    Probe parametrisation. ``v`` is held internally at unit L2 norm. When
    ``per_entry_scale`` is True (default) the probe magnitude is set so
    each entry of the perturbation is approximately ``eps`` in absolute
    value: ``p = eps * sqrt(total_dim) * v``. This is the version you want
    in bf16 / fp16 contexts, because individual cache entries have to
    change by at least one bf16 ulp (~1/128 at unit scale ~ 7.8e-3) for
    the model's forward to register the perturbation. With unit-norm ``v``
    in high dimension (e.g. ~2e5 KV-cache entries on Phi at 32k), per-
    entry magnitudes shrink as 1/sqrt(d); without the sqrt(d) rescaling
    the probe rounds to zero in bf16 and ``jv`` comes out identically
    zero. When ``per_entry_scale`` is False, ``p = eps * v`` directly --
    the original parametrisation, retained for the fp32 unit tests where
    rounding is not a concern.

    For linear operators the central difference is exact for any probe
    magnitude, so increasing the per-entry probe size to clear bf16
    rounding noise is free; the truncation error in the nonlinear case
    grows as O((eps * sqrt(d))^2) but for ``eps`` in [1e-2, 1e-1] and the
    operators we measure this remains well inside the linear regime.

    Each iteration costs 2 forward calls (JVP) + 1 forward + 1 backward
    (VJP); the final estimate adds 2 more forward calls. Returns a dict
    with ``top_sigma`` and the per-iteration JV norm history.

    The estimator is consistent: as ``eps -> 0`` and ``n_iter -> infty``
    (with a spectral gap), ``top_sigma`` converges to the operator norm of
    the Jacobian at zero. With a flat spectrum or a small spectral gap the
    estimate is a lower bound that approaches the top singular value at a
    rate set by the ratio sigma_2 / sigma_1.

    ``unit_norm_grad`` (default True) controls the VJP's grad_outputs
    magnitude: when True, the backward pass is computed with the unit
    direction of ``jv`` and the magnitude is reattached after, keeping the
    intermediate gradients in O(1) range so they survive bf16 propagation
    on models with large output norms (LLaMA's masked-decode logits norm
    around 10^3 vs Phi's 3x10^2; the original 2026-05-22 P20 LLaMA run hit
    NaN here, P20b uses this fix). Mathematically equivalent because J^T
    is linear in its grad-output argument; only the dtype range changes.

    ``detect_nan`` (default True) checks for NaN/Inf in the JVP and VJP
    outputs at each iteration; on detection, the perturbation direction
    is resampled and the iteration continues. The final top_sigma is
    reported as NaN if the FINAL JVP after all iterations still NaNs;
    callers can branch on this to drop bad cells from aggregates.
    """
    if rng is None:
        v = torch.randn(total_dim, device=device, dtype=torch.float32)
    else:
        v = torch.randn(
            total_dim, device=device, dtype=torch.float32, generator=rng,
        )
    v = v / (v.norm() + eps_floor)

    probe_scale = math.sqrt(float(total_dim)) if per_entry_scale else 1.0
    probe_norm = eps * probe_scale

    def _is_bad(t: torch.Tensor) -> bool:
        if not detect_nan:
            return False
        return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())

    def _resample(v_cur: torch.Tensor) -> torch.Tensor:
        v_new = torch.randn_like(v_cur)
        return v_new / (v_new.norm() + eps_floor)

    jv_history: list[float] = []
    n_resamples = 0
    for _ in range(n_iter):
        with torch.no_grad():
            out_plus = run_fn(probe_norm * v)
            out_minus = run_fn(-probe_norm * v)
            jv = (out_plus - out_minus) / (2.0 * probe_norm)
        if _is_bad(jv):
            jv_history.append(float("nan"))
            v = _resample(v)
            n_resamples += 1
            continue
        jv_norm = float(jv.norm().item())
        jv_history.append(jv_norm)
        if jv_norm < eps_floor:
            v = _resample(v)
            continue

        # VJP. With unit_norm_grad=True we feed J^T(jv / ||jv||) instead
        # of J^T(jv); the result is direction-equivalent (J^T is linear)
        # and the backward pass operates in O(1) gradient magnitudes
        # rather than O(||jv||), which on LLaMA's large logit norms can
        # otherwise overflow bf16 mid-backward and produce NaN.
        if unit_norm_grad:
            grad_out = jv / jv_norm
        else:
            grad_out = jv
        v_req = v.detach().clone().requires_grad_(True)
        out = run_fn(probe_norm * v_req)
        try:
            (jtjv,) = torch.autograd.grad(
                outputs=out, inputs=v_req,
                grad_outputs=grad_out.to(out.dtype),
                retain_graph=False, create_graph=False,
            )
        except RuntimeError:
            v = _resample(v)
            n_resamples += 1
            continue
        if _is_bad(jtjv):
            v = _resample(v)
            n_resamples += 1
            continue
        jtjv_norm = float(jtjv.norm().item())
        if jtjv_norm < eps_floor:
            v = _resample(v)
            continue
        v = (jtjv / jtjv_norm).detach()

    with torch.no_grad():
        out_plus = run_fn(probe_norm * v)
        out_minus = run_fn(-probe_norm * v)
        jv_final = (out_plus - out_minus) / (2.0 * probe_norm)
    if _is_bad(jv_final):
        top_sigma = float("nan")
    else:
        top_sigma = float(jv_final.norm().item())
    return {
        "top_sigma": top_sigma,
        "jv_norm_history": jv_history,
        "n_iter": n_iter,
        "n_resamples": n_resamples,
        "eps": eps,
        "probe_norm": float(probe_norm),
        "per_entry_scale": bool(per_entry_scale),
        "unit_norm_grad": bool(unit_norm_grad),
        "total_dim": int(total_dim),
    }
