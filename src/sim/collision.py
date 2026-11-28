"""Pure-JAX sliding circle-segment and circle-circle collision resolution."""

import jax
import jax.numpy as jnp


def resolve_circle_segment(
    center: jax.Array,
    radius: float,
    segment: jax.Array,
    active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Push a circle out of a single line segment along the surface normal.

    Args:
        center: (2,) agent center [x, y].
        radius: Agent collision radius.
        segment: (2, 2) endpoints [[x1, y1], [x2, y2]].
        active: Boolean scalar indicating whether this segment is active (not padding).

    Returns:
        (new_center, hit) where hit is a boolean scalar.
    """
    p1 = segment[0]
    p2 = segment[1]
    d = p2 - p1
    len_sq = jnp.dot(d, d)

    safe_len_sq = jnp.where(len_sq > 0.0, len_sq, 1.0)
    t = jnp.clip(jnp.dot(center - p1, d) / safe_len_sq, 0.0, 1.0)
    closest = p1 + t * d
    diff = center - closest
    dist = jnp.linalg.norm(diff)

    # Guard against exact-center overlap (dist <= 1e-6)
    safe_dist = jnp.maximum(dist, 1e-6)
    normal = jnp.where(dist > 1e-6, diff / safe_dist, jnp.array([1.0, 0.0], dtype=jnp.float32))

    hit = active & (len_sq > 0.0) & (dist < radius)
    new_center = jnp.where(hit, center + normal * (radius - dist), center)
    return new_center, hit


def _resolve_single_agent_walls(
    center: jax.Array,
    alive: jax.Array,
    radius: float,
    segments: jax.Array,
    segment_mask: jax.Array,
    iterations: int,
) -> tuple[jax.Array, jax.Array]:
    """Run iterative sliding wall collision passes for a single agent."""

    def one_pass(carry: tuple[jax.Array, jax.Array], _) -> tuple[tuple[jax.Array, jax.Array], None]:
        pos, collided = carry

        def seg_step(
            inner: tuple[jax.Array, jax.Array],
            seg_and_mask: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            p, hit_acc = inner
            seg, mask = seg_and_mask
            next_p, hit = resolve_circle_segment(p, radius, seg, mask & alive)
            return (next_p, hit_acc | hit), None

        (next_pos, next_hit), _ = jax.lax.scan(seg_step, (pos, collided), (segments, segment_mask))
        return (next_pos, next_hit), None

    (final_pos, any_hit), _ = jax.lax.scan(one_pass, (center, jnp.asarray(False)), None, length=iterations)
    return final_pos, any_hit


def resolve_agent_walls(
    pos: jax.Array,
    alive: jax.Array,
    radius: float,
    segments: jax.Array,
    segment_mask: jax.Array,
    iterations: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Resolve sliding wall collisions for all N agents in parallel."""
    return jax.vmap(
        _resolve_single_agent_walls,
        in_axes=(0, 0, None, None, None, None),
    )(pos, alive, radius, segments, segment_mask, iterations)


def resolve_agent_agent(
    pos: jax.Array,
    alive: jax.Array,
    radius: float,
) -> jax.Array:
    """Symmetrically separate overlapping alive agents via circle-circle push-out."""
    n = pos.shape[0]
    min_dist = 2.0 * radius

    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)

    eye = jnp.eye(n, dtype=bool)
    pair_active = (~eye) & alive[:, None] & alive[None, :]
    overlap = pair_active & (dist < min_dist)

    safe_dist = jnp.maximum(dist, 1e-6)
    fallback_normal = jnp.array([1.0, 0.0], dtype=jnp.float32)
    normal = jnp.where(
        dist[..., None] > 1e-6,
        diff / safe_dist[..., None],
        fallback_normal,
    )

    push_mag = jnp.where(overlap, 0.5 * (min_dist - dist), 0.0)
    total_push = jnp.sum(normal * push_mag[..., None], axis=1)
    return jnp.where(alive[:, None], pos + total_push, pos)
