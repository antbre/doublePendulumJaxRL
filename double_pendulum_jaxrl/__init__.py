"""Pure-JAX double-pendulum RL environment (swing-up + balance)."""

from .config import ActuationMode, EnvParams
from .env import DoublePendulum, EnvState

__all__ = ["ActuationMode", "EnvParams", "DoublePendulum", "EnvState"]
