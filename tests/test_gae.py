import numpy as np
import pytest

from deco_swarm.training.gae import compute_gae

T, B = 16, 3
GAMMA, LAM = 0.99, 0.95


def gae_numpy(rewards, values, dones, last_value, gamma, lam):
    """Reference: plain for loop, from the end to the start."""
    adv = np.zeros_like(rewards)
    gae = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
        next_value = values[t]
    return adv


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(T, B)).astype(np.float32)
    values = rng.normal(size=(T, B)).astype(np.float32)
    dones = (rng.random((T, B)) < 0.1).astype(np.float32)
    last_value = rng.normal(size=(B,)).astype(np.float32)
    return rewards, values, dones, last_value


def test_matches_numpy(batch):
    adv, targets = compute_gae(*batch, GAMMA, LAM)
    ref = gae_numpy(*batch, GAMMA, LAM)
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(targets, ref + batch[1], rtol=1e-5, atol=1e-5)


def test_lambda_zero_is_td(batch):
    rewards, values, dones, last_value = batch
    adv, _ = compute_gae(*batch, GAMMA, 0.0)
    next_values = np.concatenate([values[1:], last_value[None]])
    delta = rewards + GAMMA * next_values * (1 - dones) - values
    np.testing.assert_allclose(adv, delta, rtol=1e-5, atol=1e-5)


def test_lambda_one_is_discounted_return(batch):
    rewards, values, _, last_value = batch
    dones = np.zeros_like(rewards)
    _, targets = compute_gae(rewards, values, dones, last_value, GAMMA, 1.0)
    ret = np.zeros_like(rewards)
    g = last_value
    for t in reversed(range(T)):
        g = rewards[t] + GAMMA * g
        ret[t] = g
    np.testing.assert_allclose(targets, ret, rtol=1e-4, atol=1e-4)


def test_done_blocks_leak():
    rewards = np.zeros((4, 1), np.float32)
    rewards[3] = 100.0                      # big reward in the NEXT episode
    values = np.zeros((4, 1), np.float32)
    dones = np.zeros((4, 1), np.float32)
    dones[1] = 1.0                          # episode ends after step 1
    adv, _ = compute_gae(rewards, values, dones, np.zeros(1, np.float32), GAMMA, LAM)
    assert float(adv[0, 0]) == 0.0 and float(adv[1, 0]) == 0.0   # no leak
    assert float(adv[2, 0]) > 0.0