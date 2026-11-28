import jax.numpy as jnp

_EPS = 1e-6

def _closest_points(pos, segments):
    """pos (N, 2), segments (S, 4) -> closest point on each segment to each agent, (N, S, 2)."""
    a, b = segments[:, :2], segments[:, 2:] # (S, 2)
    ab = b - a
    ap = pos[:, None, :] - a[None, :, :] # (N, S, 2)
    t = jnp.sum(ap * ab, axis=-1) / jnp.maximum(jnp.sum(ab * ab, axis=-1), _EPS)
    t = jnp.clip(t, 0.0, 1.0) # stay on the segment
    return a + t[..., None] * ab

def push_out_of_walls(pos, alive, radius, segments, segment_mask):
    """Push each agent out of its deepest overlapping wall (one pass)."""
    closest = _closest_points(pos, segments)
    diff = pos[:, None, :] - closest
    dist = jnp.linalg.norm(diff, axis=-1)
    pen = jnp.where(segment_mask[None, :], radius - dist, 0.0)
    pen = jnp.maximum(pen, 0.0)

    k = jnp.argmax(pen, axis=1)
    idx = jnp.arange(pos.shape[0])
    depth = pen[idx, k]
    normal = diff[idx, k] / jnp.maximum(dist[idx, k], _EPS)[:, None]

    new_pos = pos + normal * depth[:, None]
    return jnp.where(alive[:, None], new_pos, pos)

def separate_agents(pos, alive, radius):
    """Push overlapping alive agents apart, half each."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    valid = alive[:, None] & alive[None, :] & ~jnp.eye(n, dtype=bool)
    overlap = jnp.where(valid, jnp.maximum(2.0 * radius - dist, 0.0), 0.0)

    idx = jnp.arange(n)
    fallback = jnp.sign(idx[:, None] - idx[None, :]).astype(pos.dtype)[..., None] * jnp.array([1.0, 0.0])
    normal = jnp.where(dist[..., None] > _EPS, diff / jnp.maximum(dist, _EPS)[..., None], fallback)

    return pos + 0.5 * jnp.sum(normal * overlap[..., None], axis=1)

def collide(pos, alive, radius, segments, segment_mask, iterations):
    """Agents first, walls last so walls always win."""
    for _ in range(iterations):
        pos = separate_agents(pos, alive, radius)
        pos = push_out_of_walls(pos, alive, radius, segments, segment_mask)
    return pos