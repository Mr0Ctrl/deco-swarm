import jax
import jax.numpy as jnp

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    dones = dones.astype(rewards.dtype)

    def step(carry, x):
        gae, next_value = carry # info coming from the "future"
        reward, value, done = x # current step t
        not_done = 1.0 - done

        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lam * not_done * gae

        return (gae, value), gae # value becomes "next_value" for step t-1
    
    init = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(step, init, (rewards, values, dones), reverse=True)
    return advantages, advantages + values # target = Â + V (what the critic learns)