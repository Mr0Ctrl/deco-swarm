"""Pure-JAX 2D kinematics and discrete action decoding."""

import jax
import jax.numpy as jnp

# Translation choices: 0=No-Op, 1=Fwd, 2=Back, 3=Strafe Left, 4=Strafe Right
_MOVE_TABLE = jnp.asarray(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ],
    dtype=jnp.float32,
)

# Rotation choices: 0=No-Op, 1=Left (-1), 2=Right (+1)
_ROT_TABLE = jnp.asarray([0.0, -1.0, 1.0], dtype=jnp.float32)


def decode_actions(
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map discrete [move(5), rotate(3), tag(2)] indices to continuous controls.

    Args:
        actions: Integer array of shape (..., 3).

    Returns:
        Tuple of (move_fwd, strafe, rot_dir, fire_tag) matching leading shape (...).
    """
    move_vec = _MOVE_TABLE[actions[..., 0]]
    rot_dir = _ROT_TABLE[actions[..., 1]]
    fire_tag = actions[..., 2] > 0
    return move_vec[..., 0], move_vec[..., 1], rot_dir, fire_tag


def step_kinematics(
    pos: jax.Array,
    heading_deg: jax.Array,
    move_fwd: jax.Array,
    strafe: jax.Array,
    rot_dir: jax.Array,
    alive: jax.Array,
    linear_speed: float,
    angular_speed: float,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Advance 2D positions and headings for alive agents over dt seconds.

    Coordinate frame: +x right, +y down; 0 deg = +x, 90 deg = +y.
    """
    new_heading = jnp.mod(heading_deg + rot_dir * angular_speed * dt, 360.0)
    new_heading = jnp.where(alive, new_heading, heading_deg)

    # Quake-style diagonal speed cap: prevent faster movement when fwd + strafe combine
    mag = jnp.hypot(move_fwd, strafe)
    scale = jnp.where(mag > 1.0, 1.0 / jnp.maximum(mag, 1e-6), 1.0)
    fwd_cmd = move_fwd * scale
    strafe_cmd = strafe * scale

    rad = jnp.deg2rad(new_heading)
    cos_h = jnp.cos(rad)
    sin_h = jnp.sin(rad)

    # Forward = (cos, sin), Right = (-sin, cos) in (+x right, +y down)
    vx = (cos_h * fwd_cmd - sin_h * strafe_cmd) * linear_speed
    vy = (sin_h * fwd_cmd + cos_h * strafe_cmd) * linear_speed
    delta = jnp.stack([vx, vy], axis=-1) * dt

    new_pos = jnp.where(alive[..., None], pos + delta, pos)
    return new_pos, new_heading
