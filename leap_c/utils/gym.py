from typing import List, Callable, TypeVar

import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics

WrapperFn = Callable[[gym.Env], gym.Env]
AnyEnv = TypeVar('AnyEnv', gym.Env, gym.vector.SyncVectorEnv)


def wrap_env(env: gym.Env, wrappers: List[WrapperFn] | None = None) -> gym.Env:
    """Wraps a gymnasium environment.

    Args:
        env: The environment to wrap.
        wrappers: A list of wrappers to apply to the environment.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = RecordEpisodeStatistics(env, buffer_length=1)
    env = OrderEnforcing(env)

    if wrappers is None:
        wrappers = []
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def seed_env(env: AnyEnv, seed: int = 0) -> AnyEnv:
    """Seeds the environment.

    Args:
        env: The environment to seed.
        seed: The seed to use.

    Returns:
        AnyEnv: The seeded environment.
    """
    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env
