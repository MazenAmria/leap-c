from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from leap_c.autograd.function import DiffFunction
from leap_c.autograd.jax import create_jax_callable


class DummyFunction(DiffFunction):
    """Simple test function with known analytical derivatives.

    Forward returns: u_star = x0 + 1, value = sum(x0**2).
    Backward returns analytical VJP.
    """

    def forward(self, ctx, x0_np, u0_np, p_global_np, p_stagewise_np, p_stagewise_sparse_idx_np):
        B = x0_np.shape[0]
        ctx = SimpleNamespace(x0=x0_np.copy())
        u_star = x0_np + 1
        x = np.broadcast_to(u_star[:, None, :], (B, 3, x0_np.shape[1]))
        u = np.broadcast_to(u_star[:, None, :], (B, 2, x0_np.shape[1]))
        value = np.sum(x0_np**2, axis=1, keepdims=True)
        return ctx, u_star, x, u, value

    def backward(self, ctx, u0_grad, x_grad, u_grad, value_grad):
        x0_np = ctx.x0
        grad_x0 = np.zeros_like(x0_np)
        if u0_grad is not None:
            grad_x0 += u0_grad
        if value_grad is not None:
            grad_x0 += 2 * x0_np * value_grad
        return grad_x0, None, None, None, None


def test_create_jax_callable_forward():
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2, np_global=1)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.0], [0.0]])
    p_stagewise = jnp.zeros((2, 3, 1))

    u_star, x, u, value = solve(x0, u0, p_global, p_stagewise)

    expected_u_star = x0 + 1
    expected_value = jnp.sum(x0**2, axis=1, keepdims=True)

    assert jnp.allclose(u_star, expected_u_star)
    assert jnp.allclose(value, expected_value)


def test_create_jax_callable_grad_wrt_x0():
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2, np_global=1)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.0], [0.0]])
    p_stagewise = jnp.zeros((2, 3, 1))

    def loss_fn(x0_val):
        u_star, x, u, value = solve(x0_val, u0, p_global, p_stagewise)
        return jnp.sum(u_star) + jnp.sum(value)

    grad_x0 = jax.grad(loss_fn)(x0)
    expected_grad = jnp.ones_like(x0) + 2 * x0
    assert jnp.allclose(grad_x0, expected_grad, atol=1e-6)


def test_create_jax_callable_grad_wrt_p_global():
    class DummyWithP(DiffFunction):
        def forward(
            self, ctx, x0_np, u0_np, p_global_np, p_stagewise_np, p_stagewise_sparse_idx_np
        ):
            B = x0_np.shape[0]
            ctx = SimpleNamespace(x0=x0_np.copy(), p_global=p_global_np.copy())
            u_star = x0_np + p_global_np[:, :2]
            x = np.broadcast_to(u_star[:, None, :], (B, 3, 2))
            u = np.broadcast_to(u_star[:, None, :], (B, 2, 2))
            value = np.sum(p_global_np**2, axis=1, keepdims=True)
            return ctx, u_star, x, u, value

        def backward(self, ctx, u0_grad, x_grad, u_grad, value_grad):
            x0_np = ctx.x0
            p_global_np = ctx.p_global
            grad_x0 = np.zeros_like(x0_np)
            grad_p_global = np.zeros_like(p_global_np)
            if u0_grad is not None:
                grad_x0 += u0_grad
                grad_p_global[:, :2] += u0_grad
            if value_grad is not None:
                grad_p_global += 2 * p_global_np * value_grad
            return grad_x0, None, grad_p_global, None, None

    fun = DummyWithP()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2, np_global=3)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    p_stagewise = jnp.zeros((2, 3, 1))

    def loss_fn(p_val):
        u_star, x, u, value = solve(x0, u0, p_val, p_stagewise)
        return jnp.sum(u_star) + jnp.sum(value)

    grad_p = jax.grad(loss_fn)(p_global)
    # d(u_star)/dp = [1, 1, 0], d(value)/dp = 2*p
    # grad_p[:, :2] = 1 + 2*p[:, :2], grad_p[:, 2] = 2*p[:, 2]
    expected_grad_p = jnp.zeros_like(p_global)
    expected_grad_p = expected_grad_p.at[:, :2].set(
        jnp.ones_like(p_global[:, :2]) + 2 * p_global[:, :2]
    )
    expected_grad_p = expected_grad_p.at[:, 2].set(2 * p_global[:, 2])

    assert jnp.allclose(grad_p, expected_grad_p, atol=1e-6)
