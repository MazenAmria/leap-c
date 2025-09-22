import torch


class GeneralizedAdvantageEstimation:
    def __init__(self, gamma: float, lambd: float):
        self.gamma = gamma
        self.lambd = lambd

    def calculate(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminations: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pad the values with a zero at the end
        zeros = torch.zeros((1, *values.shape[1:]), device=values.device, dtype=values.dtype)
        values = torch.cat((values, zeros), dim=0)

        # TD residual: δ = r + γ * V' - V
        deltas = rewards + self.gamma * values[1:] * (1.0 - terminations) - values[:-1]

        # initialize advantages and returns
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        # GAE: A = δ + γ * λ * A'
        for t in reversed(range(values.shape[0] - 1)):
            advantages[t] = (
                deltas[t] + self.gamma * self.lambd * (1.0 - dones[t]) * advantages[t + 1]
            )

        returns = advantages + values

        return advantages[:-1], returns[:-1]
