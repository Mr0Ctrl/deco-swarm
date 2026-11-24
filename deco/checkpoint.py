# deco/checkpoint.py
"""
Checkpoint save/load (Flax msgpack).

A checkpoint stores the network parameters, the optimizer state and metadata
(step, curriculum tier/map, success rate, average steps, seed, configs).
`load_checkpoint` also accepts legacy PyTorch .pt files (or their extracted directories) and converts them.
"""
import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import serialization

from deco.torch_import import load_torch_checkpoint, torch_state_dict_to_flax

META_KEYS = ("step", "tier_idx", "map_idx", "success_rate", "avg_steps")


def save_checkpoint(path: str, params, opt_state=None, **meta) -> None:
    payload = {
        "params": serialization.to_state_dict(jax.device_get(params)),
        "opt_state": serialization.to_state_dict(jax.device_get(opt_state)) if opt_state is not None else None,
        "meta": {k: v for k, v in meta.items() if v is not None},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(serialization.msgpack_serialize(payload))
    os.replace(tmp_path, path)  # atomic: an interrupted save never leaves a half-written checkpoint


def _is_torch_checkpoint(path: str) -> bool:
    return path.endswith(".pt") or path.endswith(".pth") or os.path.isdir(path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Returns {"params": flax params, "opt_state": state dict or None, "meta": {...}}."""
    if _is_torch_checkpoint(path):
        raw = load_torch_checkpoint(path)
        state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
        meta = {k: raw[k] for k in META_KEYS if isinstance(raw, dict) and k in raw}
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}
        params = torch_state_dict_to_flax(state_dict)
        return {"params": jax.tree.map(jnp.asarray, params), "opt_state": None, "meta": meta}

    with open(path, "rb") as f:
        payload = serialization.msgpack_restore(f.read())
    return {
        "params": jax.tree.map(jnp.asarray, payload["params"]),
        "opt_state": payload.get("opt_state"),
        "meta": payload.get("meta") or {},
    }
