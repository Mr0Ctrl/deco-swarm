import jax.numpy as jnp

from deco_swarm.config.schema import BodyConfig

# action index -> sign. 0 = none, 1 = +, 2 = -
_SIGN = jnp.array([0.0, 1.0, -1.0])

def decode_actions(actions):
    """actions: (N, 4) int [move, strafe, turn, fire] -> (fwd, left, turn, fire), each (N,)."""
    fwd = _SIGN[actions[:, 0]] # +1 forward, -1 back
    left = _SIGN[actions[:, 1]] # +1 left, -1 right
    turn = _SIGN[actions[:, 2]] # +1 CCW, -1 CW
    fire = actions[:, 3] == 1
    return fwd, left, turn, fire


def integrate(pos, heading_deg, actions, alive, body: BodyConfig, dt):
    """One physics tick of movement, before collisions.

    pos: (N, 2), heading_deg: (N,), actions: (N, 4) int, alive: (N,) bool.
    Returns new (pos, heading_deg).
    """
    fwd, left, turn, _ = decode_actions(actions)

    # 1) turn first, keep the angle in [0, 360)
    heading = jnp.mod(heading_deg + turn * body.angular_speed_dps * dt, 360.0)

    # 2) local move direction; normalize so diagonals are not faster
    norm = jnp.sqrt(fwd**2 + left**2)
    scale = jnp.where(norm > 1.0, 1.0 / jnp.maximum(norm, 1e-6), 1.0)
    fwd, left = fwd * scale, left * scale

    # 3) local -> world: forward and left unit vectors from the heading
    h = jnp.deg2rad(heading)
    forward_vec = jnp.stack([jnp.cos(h), jnp.sin(h)], axis=-1) # +x at 0 deg
    left_vec = jnp.stack([-jnp.sin(h), jnp.cos(h)], axis=-1) # 90 deg CCW of forward
    vel = (fwd[:, None] * forward_vec + left[:, None] * left_vec) * body.linear_speed_mps

    # 4) dead agents stay put
    new_pos = pos + vel * dt
    pos = jnp.where(alive[:, None], new_pos, pos)
    heading = jnp.where(alive, heading, heading_deg)
    return pos, heading