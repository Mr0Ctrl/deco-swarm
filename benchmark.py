# benchmark.py
"""
Throughput benchmark.

  python benchmark.py                          # simulator only, several batch sizes
  python benchmark.py --train                  # also full PPO iterations (rollout + update)
  python benchmark.py --num-envs 256,1024,4096 --device gpu

Reports environment decision steps per second (each step = action_repeat physics ticks + raycast).
"""
import argparse
import time

import jax
import jax.numpy as jnp

from deco.config import load_configs
from deco.curriculum import MapCurriculum
from deco.env import MapPool, NavEnv
from deco.maps import MapBank
from deco.model import RecurrentActorCritic
from deco.ppo import PPOConfig, PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DECO-SWARM throughput benchmark")
    parser.add_argument("--num-envs", type=str, default="64,256,1024", help="Comma-separated batch sizes")
    parser.add_argument("--steps", type=int, default=256, help="Decision steps per measurement")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions (after compilation)")
    parser.add_argument("--train", action="store_true", help="Also benchmark full PPO iterations")
    parser.add_argument("--device", type=str, default="auto", help="gpu, cpu or auto")
    return parser.parse_args()


def bench_env(env: NavEnv, pool: MapPool, num_envs: int, steps: int, repeats: int) -> float:
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run(key):
        k_reset, k_run = jax.random.split(key)
        _, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, num_envs), pool)

        def body(carry, _):
            state, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            actions = jax.random.randint(k_act, (num_envs, 2), 0, jnp.array([5, 3]))
            _, state, reward, _, _ = step(jax.random.split(k_env, num_envs), state, actions, pool)
            return (state, key), reward

        _, rewards = jax.lax.scan(body, (state, k_run), None, length=steps)
        return rewards.sum()

    run(jax.random.PRNGKey(0)).block_until_ready()  # compile
    best = float("inf")
    for r in range(repeats):
        t0 = time.perf_counter()
        run(jax.random.PRNGKey(r + 1)).block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return num_envs * steps / best


def bench_train(env: NavEnv, pool: MapPool, num_envs: int, steps: int, repeats: int) -> float:
    model = RecurrentActorCritic()
    trainer = PPOTrainer(env, model, PPOConfig(num_envs=num_envs, rollout_steps=steps))
    runner = trainer.init_runner(jax.random.PRNGKey(0), trainer.init_params(jax.random.PRNGKey(1)), pool)

    def iteration(runner):
        runner, rollout, _ = trainer.collect(runner, pool)
        runner, metrics = trainer.train(runner, rollout)
        jax.block_until_ready(metrics)
        return runner

    runner = iteration(runner)  # compile
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        runner = iteration(runner)
        best = min(best, time.perf_counter() - t0)
    return num_envs * steps / best


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif args.device in ("gpu", "cuda"):
        jax.config.update("jax_platforms", "cuda")

    print(f"JAX {jax.__version__} | backend: {jax.default_backend()} | device: {jax.devices()[0].device_kind}")

    curriculum = MapCurriculum()
    paths = curriculum.all_map_paths()
    bank = MapBank(paths)
    env = NavEnv(load_configs(), bank)
    pool = MapPool(*map(jnp.asarray, bank.pool(paths)))
    print(f"{len(bank)} maps (up to {bank.segments.shape[1]} wall segments), {env.agent.ray_count} rays, "
          f"{env.cfg.action_repeat} physics ticks per step\n")

    for n in (int(x) for x in args.num_envs.split(",")):
        sps = bench_env(env, pool, n, args.steps, args.repeats)
        line = f"num_envs={n:6d} | simulator: {sps:12,.0f} steps/s"
        if args.train:
            line += f" | training (rollout + PPO update): {bench_train(env, pool, n, args.steps, args.repeats):10,.0f} steps/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
