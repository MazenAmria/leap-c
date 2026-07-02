"""Creates JAX callables with custom VJP rules from DiffFunction objects."""

from typing import Callable

import numpy as np
from jax import custom_vjp, pure_callback
from jax import numpy as jnp
from jax.core import ShapedArray

from leap_c.autograd.function import DiffFunction
from leap_c.utils.dependencies import require_jax

jax = require_jax()


def _to_np(val):
    if val is None:
        return None
    return np.asarray(val)


def _to_np_tuple(*vals):
    return tuple(_to_np(v) for v in vals)


def create_jax_callable(
    fun: DiffFunction,
    N_horizon: int,
    nx: int,
    nu: int,
    np_global: int,
) -> Callable:
    """Creates a JAX-callable function with a custom VJP wrapping a ``DiffFunction``.

    The returned function accepts ``(x0, u0, p_global, p_stagewise)`` and returns
    ``(sol_u0, x, u, sol_value)``.  All inputs and outputs are JAX arrays, except
    ``u0`` which may also be a Python ``None``.

    The impure acados C solver is isolated inside ``jax.pure_callback``, and
    gradients flow via ``jax.custom_vjp``.  The backward pass stores forward-pass
    context in a closure, so ``jax.grad`` works but ``jax.jit(jax.grad(...))`` is
    not supported (context can not cross the XLA compilation boundary).

    ``pure_callback`` passes JAX arrays (not numpy) to the callback, so explicit
    ``np.asarray()`` conversion is performed before calling the core solver.

    Args:
        fun: Framework-agnostic ``DiffFunction`` (e.g., ``AcadosDiffMpcFunction``).
        N_horizon: OCP horizon length.
        nx: State dimension.
        nu: Control dimension.
        np_global: Size of the flat differentiable parameter vector.

    Returns:
        A JAX-callable function whose gradient is defined by ``fun.backward``.
    """
    _forward_ctx = None

    def _result_shapes(B, dtype):
        return (
            ShapedArray((B, nu), dtype),
            ShapedArray((B, N_horizon + 1, nx), dtype),
            ShapedArray((B, N_horizon, nu), dtype),
            ShapedArray((B, 1), dtype),
        )

    @custom_vjp
    def solve(x0, u0, p_global, p_stagewise):
        B = x0.shape[0]
        dtype = x0.dtype
        result_shapes = _result_shapes(B, dtype)

        def callback(x0_jax, u0_jax, p_global_jax, p_stagewise_jax):
            x0_np, u0_np, p_global_np, p_stagewise_np = _to_np_tuple(
                x0_jax, u0_jax, p_global_jax, p_stagewise_jax
            )
            _, u_star, x, u, value = fun.forward(
                None, x0_np, u0_np, p_global_np, p_stagewise_np, None
            )
            return u_star, x, u, value

        return pure_callback(callback, result_shapes, x0, u0, p_global, p_stagewise)

    def solve_fwd(x0, u0, p_global, p_stagewise):
        B = x0.shape[0]
        dtype = x0.dtype
        result_shapes = _result_shapes(B, dtype)

        def callback(x0_jax, u0_jax, p_global_jax, p_stagewise_jax):
            nonlocal _forward_ctx
            x0_np, u0_np, p_global_np, p_stagewise_np = _to_np_tuple(
                x0_jax, u0_jax, p_global_jax, p_stagewise_jax
            )
            ctx, u_star, x, u, value = fun.forward(
                None, x0_np, u0_np, p_global_np, p_stagewise_np, None
            )
            _forward_ctx = ctx
            return u_star, x, u, value

        u_star, x, u, value = pure_callback(callback, result_shapes, x0, u0, p_global, p_stagewise)
        return (u_star, x, u, value), None

    def solve_bwd(res, *cts):
        nonlocal _forward_ctx
        if _forward_ctx is None:
            return (None,) * 4

        _forward_ctx.needs_input_grad = (False, True, True, True, False, False)

        cts = cts[0] if len(cts) == 1 and isinstance(cts[0], (tuple, list)) else cts

        ct_u_star = cts[0] if len(cts) > 0 else None
        ct_x = cts[1] if len(cts) > 1 else None
        ct_u = cts[2] if len(cts) > 2 else None
        ct_value = cts[3] if len(cts) > 3 else None

        np_ct = tuple(
            None if ct is None else np.asarray(ct) for ct in (ct_u_star, ct_x, ct_u, ct_value)
        )
        grad_x0, grad_u0, grad_p_global, _, _ = fun.backward(_forward_ctx, *np_ct)

        def _to_jax(arr):
            return None if arr is None else jnp.asarray(arr)

        return _to_jax(grad_x0), _to_jax(grad_u0), _to_jax(grad_p_global), None

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
