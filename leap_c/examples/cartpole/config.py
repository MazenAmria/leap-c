from dataclasses import dataclass
from leap_c.ocp.acados.parameters import Parameter, AcadosParamManager
import numpy as np


@dataclass(kw_only=True)
class CartPoleParams:
    # Dynamics parameters
    M: Parameter  # mass of the cart [kg]
    m: Parameter  # mass of the ball [kg]
    g: Parameter  # gravity constant [m/s^2]
    l: Parameter  # length of the rod [m]

    # Cost matrix factorization parameters
    q_diag: Parameter
    r_diag: Parameter

    # Reference parameters (for NONLINEAR_LS cost)
    xref1: Parameter  # reference position
    xref2: Parameter  # reference theta
    xref3: Parameter  # reference v
    xref4: Parameter  # reference thetadot
    uref: Parameter  # reference u


def make_default_cartpole_params(stage_wise: bool = False) -> CartPoleParams:
    """Returns a CartPoleParams instance with default parameter values."""
    return CartPoleParams(
        # Dynamics parameters
        M=Parameter("M", np.array([1.0])),
        m=Parameter("m", np.array([0.1])),
        g=Parameter("g", np.array([9.81])),
        l=Parameter("l", np.array([0.8])),
        # Cost matrix factorization parameters
        q_diag=Parameter("q_diag", np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2]))),
        r_diag=Parameter("r_diag", np.sqrt(np.array([2e-1]))),
        # Reference parameters (for NONLINEAR_LS cost)
        xref1=Parameter("xref1", np.array([0.0])),
        xref2=Parameter(
            "xref2",
            np.array([0.0]),
            lower_bound=-2.0 * np.pi,
            upper_bound=2.0 * np.pi,
            differentiable=True,
            stage_wise=stage_wise,
        ),
        xref3=Parameter("xref3", np.array([0.0])),
        xref4=Parameter("xref4", np.array([0.0])),
        uref=Parameter("uref", np.array([0.0])),
    )
