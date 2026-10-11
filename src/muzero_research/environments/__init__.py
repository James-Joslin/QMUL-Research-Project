"""Lazy environment registry with explicit observation and reward conventions."""

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np

from ..config import EnvironmentConfig
from ..contracts import Game, Transition

ENVIRONMENTS = (
    "gridworld",
    "lunarlander",
    "lunarlander_standard",
    "cartpole",
    "atari",
    "breakout",
    "connect4",
    "gomoku",
    "tictactoe",
    "twentyone",
    "simple_grid",
    "spiel",
)


class GymGame(Game):
    def __init__(self, environment, config: EnvironmentConfig):
        self.env = environment
        self.config = config

    def _observation(self, observation):
        if isinstance(observation, dict):
            observation = observation["image"]
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim == 1:
            observation = observation.reshape(1, 1, -1)
        elif self.config.name == "gridworld" and self.config.observation_layout == "CHW":
            observation = observation.transpose(2, 0, 1)
        if observation.shape != self.config.observation_shape:
            raise ValueError(
                f"Observation shape {observation.shape} does not match {self.config.observation_shape}"
            )
        return np.ascontiguousarray(observation)

    def reset(self, *, seed=None):
        observation, _ = self.env.reset(seed=seed)
        return self._observation(observation)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return Transition(
            self._observation(observation),
            float(reward) * self.config.reward_scale,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def legal_actions(self):
        return list(self.config.action_space)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class OptionalGame(Game):
    """Bridge the retained board-game implementations into the modern episode contract."""

    def __init__(self, game, config):
        self.game = game
        self.config = config
        self.steps = 0

    def reset(self, *, seed=None):
        # The retained implementations use NumPy's RNG; reset it deliberately.
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        return np.asarray(self.game.reset(), dtype=np.float32)

    def step(self, action):
        observation, reward, done = self.game.step(action)
        self.steps += 1
        return Transition(
            np.asarray(observation, dtype=np.float32),
            float(reward) * self.config.reward_scale,
            float(reward),
            bool(done),
            not done and self.steps >= self.config.max_steps,
        )

    def to_play(self):
        return self.game.to_play()

    def legal_actions(self):
        return self.game.legal_actions()

    def render(self):
        return self.game.render()

    def close(self):
        self.game.close()


def make_game(config: EnvironmentConfig, *, render_mode: str | None = None) -> Game:
    if config.name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {config.name}; choose from {ENVIRONMENTS}")
    if config.name in {"connect4", "gomoku", "tictactoe", "twentyone", "simple_grid", "spiel"}:
        try:
            module = importlib.import_module(f".optional.{config.name}", __name__)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"Optional integration {config.name} requires its upstream dependency; see docs/experiments.md"
            ) from error
        return OptionalGame(module.Game(), config)
    import gymnasium as gym

    if config.name == "lunarlander":
        from .research_lander import DeterministicLunarLander

        env = gym.wrappers.TimeLimit(
            DeterministicLunarLander(render_mode=render_mode), max_episode_steps=config.max_steps
        )
    elif config.name == "gridworld":
        import minigrid  # noqa: F401 — registers the MiniGrid environments

        env = gym.make(
            "MiniGrid-Empty-Random-6x6-v0",
            render_mode=render_mode,
            max_episode_steps=config.max_steps,
        )
    elif config.name == "lunarlander_standard":
        env = gym.make(
            "LunarLander-v3", render_mode=render_mode, max_episode_steps=config.max_steps
        )
    elif config.name == "cartpole":
        env = gym.make("CartPole-v1", render_mode=render_mode, max_episode_steps=config.max_steps)
    else:
        try:
            import ale_py

            gym.register_envs(ale_py)
            env = gym.make(
                "ALE/Breakout-v5",
                render_mode=render_mode,
                max_episode_steps=config.max_steps,
                frameskip=1,
            )
            env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, screen_size=96)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.asarray(obs, dtype=np.float32)[None] / 255,
                gym.spaces.Box(0, 1, (1, 96, 96), dtype=np.float32),
            )
            config = replace(config, observation_shape=(1, 96, 96))
        except (ImportError, gym.error.Error) as error:
            raise RuntimeError(
                "Install the games extra and supply Atari ROMs for Breakout"
            ) from error
    return GymGame(env, config)
