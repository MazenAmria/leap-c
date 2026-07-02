import jax
import jax.numpy as jnp
import numpy as np
import pytest

from leap_c.jax import AcadosDiffMpcLayerJax

jax.config.update("jax_enable_x64", True)


def test_initialization(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    assert diff_mpc_jax is not None, "diff_mpc_jax should not be None after initialization."


def test_forward_no_params(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    x0 = jnp.ones((B, nx))
    u_star, x, u, value = diff_mpc_jax(x0)

    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)


def test_forward_with_u0(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    x0 = jnp.ones((B, nx))
    u0 = jnp.zeros((B, nu))
    u_star, x, u, value = diff_mpc_jax(x0, u0=u0)

    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)


def test_statelessness(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))
    u_star1, x1, u1, value1 = diff_mpc_jax(x0)
    u_star2, x2, u2, value2 = diff_mpc_jax(x0)

    assert jnp.allclose(u_star1, u_star2)
    assert jnp.allclose(value1, value2)


def test_grad_wrt_x0(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))

    def loss_fn(x0_val):
        u_star, x, u, value = diff_mpc_jax(x0_val)
        return jnp.sum(value)

    grad_x0 = jax.grad(loss_fn)(x0)
    assert grad_x0.shape == (B, nx)
    assert jnp.any(grad_x0 != 0), "Gradient should be non-zero"


def test_grad_wrt_u0(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu

    x0 = jnp.ones((B, nx))
    u0 = jnp.ones((B, nu))

    def loss_fn(u0_val):
        u_star, x, u, value = diff_mpc_jax(x0, u0=u0_val)
        return jnp.sum(u_star)

    grad_u0 = jax.grad(loss_fn)(u0)
    assert grad_u0.shape == (B, nu)


def test_grad_wrt_params(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    pm = diff_mpc_jax.parameter_manager
    diff_names = pm.differentiable_parameter_names
    if not diff_names:
        pytest.skip("No differentiable parameters registered")
    param_name = diff_names[0]
    default_val = pm.parameters[param_name].broadcasted_default(pm.N_horizon)

    x0 = jnp.ones((B, nx))
    param_val = jnp.array([default_val, default_val])

    def loss_fn(p_val):
        u_star, x, u, value = diff_mpc_jax(x0, params={param_name: p_val})
        return jnp.sum(value)

    grad_p = jax.grad(loss_fn)(param_val)
    assert grad_p.shape == param_val.shape


def test_forward_with_params_dict(diff_mpc_jax: AcadosDiffMpcLayerJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    pm = diff_mpc_jax.parameter_manager
    params = {}
    for name in pm.non_differentiable_parameter_names:
        default = pm.parameters[name].broadcasted_default(pm.N_horizon)
        params[name] = np.broadcast_to(default, (B, pm.N_horizon + 1, *default.shape))

    x0 = jnp.ones((B, nx))
    u_star, x, u, value = diff_mpc_jax(x0, params=params)

    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)
