"""Pure-JAX 2D headless simulation primitives."""

from .collision import resolve_agent_agent, resolve_agent_walls, resolve_circle_segment
from .ctf_world import CTFState, StepEvents, get_observations, reset_world, step_world
from .geometry import decode_actions, step_kinematics
from .raycast import cast_all_rays, compute_ego_scalars

__all__ = [
    "CTFState",
    "StepEvents",
    "cast_all_rays",
    "compute_ego_scalars",
    "decode_actions",
    "get_observations",
    "reset_world",
    "resolve_agent_agent",
    "resolve_agent_walls",
    "resolve_circle_segment",
    "step_kinematics",
    "step_world",
]
