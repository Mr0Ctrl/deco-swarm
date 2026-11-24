"""
The Flax model with converted weights must reproduce the legacy PyTorch model's outputs.
Also checks the torch-free .pt reader against torch.load. Skipped when PyTorch is not installed.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from conftest import add_legacy_to_path  # noqa: E402
from deco.checkpoint import load_checkpoint  # noqa: E402
from deco.model import RecurrentActorCritic  # noqa: E402
from deco.torch_import import load_torch_checkpoint  # noqa: E402

OLD_CHECKPOINT = "checkpoints/model_step_270000.pt"


def _torch_model():
    add_legacy_to_path()
    from models.actor_critic import RecurrentActorCritic as TorchActorCritic

    return TorchActorCritic().eval()


def _compare(torch_model, flax_params):
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(6, 640)).astype(np.float32)
    hidden = rng.normal(0, 0.5, size=(6, 128)).astype(np.float32)
    with torch.no_grad():
        t_move, t_rot, t_val, t_hidden = torch_model.forward_step(torch.from_numpy(obs), torch.from_numpy(hidden))
    f_hidden, f_move, f_rot, f_val = RecurrentActorCritic().apply(flax_params, jnp.asarray(hidden), jnp.asarray(obs))
    np.testing.assert_allclose(np.asarray(f_hidden), t_hidden.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_move), t_move.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_rot), t_rot.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_val), t_val.numpy()[:, 0], atol=1e-5)


def test_converted_weights_match_torch(tmp_path):
    torch.manual_seed(0)
    model = _torch_model()
    path = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"step": 123, "model_state_dict": model.state_dict(), "tier_idx": 1}, path)

    ckpt = load_checkpoint(path)
    assert ckpt["meta"]["step"] == 123 and ckpt["meta"]["tier_idx"] == 1
    _compare(model, ckpt["params"])


def test_torch_free_reader_matches_torch_load(tmp_path):
    torch.manual_seed(1)
    model = _torch_model()
    optimizer = torch.optim.Adam(model.parameters())
    move_logits, _, _, _ = model.forward_step(torch.zeros(1, 640), model.initial_hidden_state(1))
    move_logits.sum().backward()
    optimizer.step()  # fills the optimizer state with tensors (exp_avg, exp_avg_sq, step)
    path = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, path)

    ours = load_torch_checkpoint(path)["model_state_dict"]
    ref = torch.load(path, weights_only=True)["model_state_dict"]
    assert list(ours.keys()) == list(ref.keys())
    for k in ref:
        np.testing.assert_array_equal(ours[k], ref[k].numpy())


@pytest.mark.skipif(not os.path.exists(OLD_CHECKPOINT), reason="trained legacy checkpoint not present")
def test_trained_legacy_checkpoint():
    model = _torch_model()
    model.load_state_dict(torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"])
    _compare(model, load_checkpoint(OLD_CHECKPOINT)["params"])
