"""Pure-JAX 2D headless simulation primitives."""

from .collision import resolve_agent_agent, resolve_agent_walls, resolve_circle_segment
from .geometry import decode_actions, step_kinematics

__all__ = [
    "decode_actions",
    "resolve_agent_agent",
    "resolve_agent_walls",
    "resolve_circle_segment",
    "step_kinematics",
]
