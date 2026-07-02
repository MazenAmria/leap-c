"""Central interface to use acados in JAX."""

from pathlib import Path

import numpy as np
from acados_template import AcadosOcp

from leap_c.autograd.jax import create_jax_callable
from leap_c.diff_mpc.function import AcadosDiffMpcFunction
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters.base import AcadosParameterManager
from leap_c.repr import (
    format_diff_mpc_module_extra_repr,
    format_diff_mpc_module_repr,
)
from leap_c.utils.dependencies import require_jax

jax = require_jax()
import jax.numpy as jnp  # noqa: E402


class AcadosDiffMpcLayerJax:
    """JAX wrapper for differentiable MPC based on acados.

    This class wraps acados solvers to enable their use in differentiable JAX
    pipelines.  It uses ``jax.custom_vjp`` together with ``jax.pure_callback``
    to isolate the impure C solver.  The resulting callable is compatible with
    ``jax.grad`` (but not ``jax.jit``-of-``jax.grad`` — see :mod:`leap_c.autograd.jax`).

    Accepts a plain ``AcadosOcp`` together with a parameter manager.  The manager's
    :meth:`~AcadosParameterManager.combine_differentiable_parameters_jax` is called
    in the forward pass to build a flat differentiable array from the ``params`` dict.

    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        parameter_manager: The parameter manager instance.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    parameter_manager: AcadosParameterManager

    def __init__(
        self,
        ocp: AcadosOcp,
        parameter_manager: AcadosParameterManager,
        initializer: AcadosDiffMpcInitializer | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcLayerJax module.

        Calls ``parameter_manager.assign_to_ocp(ocp)`` to synchronise CasADi symbols
        and default values onto the OCP, then creates the solvers.

        Args:
            ocp: The acados OCP object.  Must not yet have ``model.p`` /
                ``model.p_global`` set (they will be set by ``assign_to_ocp``).
            parameter_manager: A parameter manager with registered parameters.
            initializer: The initializer used to provide initial guesses for the
                solver. Uses a zero iterate by default.
            discount_factor: An optional discount factor for the sensitivity problem.
            export_directory: An optional directory for generated C code.
            n_batch_init: Initially supported batch size.  If ``None``, a default
                is used.
            num_threads_batch_solver: Number of parallel threads for the batch solver.
            verbose: Whether to print solver generation output.
        """
        parameter_manager.assign_to_ocp(ocp)
        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.parameter_manager = parameter_manager

        np_global = self.parameter_manager.differentiable_default_flat.size
        self._solve_fn = create_jax_callable(
            self.diff_mpc_fun,
            N_horizon=ocp.solver_options.N_horizon,
            nx=ocp.dims.nx,
            nu=ocp.dims.nu,
            np_global=np_global,
        )

    def __call__(
        self,
        x0: jax.Array,
        u0: jax.Array | None = None,
        params: dict[str, jax.Array | np.ndarray] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Performs the forward pass by solving the provided problem instances.

        Builds flat ``p_global`` and ``p_stagewise`` arrays from the ``params`` dict
        using the parameter manager, then calls the underlying differentiable MPC
        function.  Gradients flow back through ``p_global`` to the individual parameter
        arrays provided in ``params``.

        Args:
            x0: Initial states with shape ``(B, x_dim)``.
            u0: Initial actions with shape ``(B, u_dim)``. Defaults to zeros.
            params: A dictionary containing named parameter overrides.  Values may be
                JAX arrays (differentiable) or numpy arrays (non-differentiable).

        Returns:
            u_star: Solution of initial control input, shape ``(B, u_dim)``.
            x: State trajectory, shape ``(B, N+1, x_dim)``.
            u: Control trajectory, shape ``(B, N, u_dim)``.
            value: Cost value, shape ``(B, 1)``.
        """
        batch_size = x0.shape[0]
        nu = self.diff_mpc_fun.ocp.dims.nu

        if u0 is None:
            u0 = jnp.zeros((batch_size, nu))

        differentiable_overwrites: dict[str, jax.Array | np.ndarray] = {}
        non_differentiable_overwrites: dict[str, np.ndarray] = {}
        if params is not None:
            for name in self.parameter_manager.differentiable_parameter_names:
                if name in params:
                    differentiable_overwrites[name] = params[name]
            for name in self.parameter_manager.non_differentiable_parameter_names:
                if name in params:
                    val = params[name]
                    if isinstance(val, np.ndarray):
                        pass
                    else:
                        val = np.asarray(val)
                    non_differentiable_overwrites[name] = val

        p_global = self.parameter_manager.combine_differentiable_parameters_jax(
            batch_size=batch_size,
            **differentiable_overwrites,
        )

        p_stagewise_np = self.parameter_manager.combine_non_differentiable_parameters(
            batch_size=batch_size, **non_differentiable_overwrites
        )
        p_stagewise = jnp.asarray(p_stagewise_np)

        return self._solve_fn(x0, u0, p_global, p_stagewise)

    def extra_repr(self) -> str:
        return format_diff_mpc_module_extra_repr(
            ocp=self.diff_mpc_fun.ocp, parameter_manager=self.parameter_manager
        )

    def __repr__(self) -> str:
        return format_diff_mpc_module_repr(
            class_name=type(self).__name__,
            ocp=self.diff_mpc_fun.ocp,
            parameter_manager=self.parameter_manager,
        )
