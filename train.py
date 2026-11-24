# train.py
"""
DECO-SWARM: Phase 1 navigation training (JAX).

Same training features as the original PyTorch script: sub-curriculum with promotion/demotion,
graduation checkpoints, tier-scoped best models, catastrophic-collapse rollback with LR annealing,
periodic checkpoints, resume (also from legacy .pt files) and tier forcing/locking.
New: experience is collected from --num-envs environments in parallel inside a jitted rollout.
"""
import argparse
import dataclasses
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from deco.checkpoint import load_checkpoint, save_checkpoint
from deco.config import load_configs
from deco.curriculum import MapCurriculum
from deco.env import MapPool, NavEnv
from deco.maps import MapBank
from deco.model import RecurrentActorCritic
from deco.ppo import EpisodeStats, PPOConfig, PPOTrainer, get_learning_rate, set_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description="DECO-SWARM: Phase 1 Navigation Training (JAX)")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total environment steps to train")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per environment per rollout")
    parser.add_argument("--batch-seq-len", type=int, default=32, help="Sequence length for recurrent unrolling")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient for exploration")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--save-interval", type=int, default=10000, help="Timesteps between saving checkpoints")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume from (.msgpack, or legacy .pt)")
    parser.add_argument("--tier", type=int, default=-1, help="Force curriculum tier (0 for faz_1v0.*, 1 for faz_1v1.*, etc.)")
    parser.add_argument("--lock-tier", action="store_true", help="Lock the chosen tier (prevent promotion/demotion)")
    parser.add_argument("--device", type=str, default="auto", help="Device: gpu, cpu, or auto")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--num-minibatches", type=int, default=8, help="Minibatches per PPO epoch")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO epochs per rollout")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (same seed + config = same run)")
    return parser.parse_args()


def record_finished_episodes(curriculum: MapCurriculum, bank: MapBank, stats: EpisodeStats):
    """Feeds episodes that ended during the rollout to the curriculum, in time order."""
    level = curriculum.level_key()
    done = np.asarray(stats.done)
    success, steps, map_idx = np.asarray(stats.success), np.asarray(stats.steps), np.asarray(stats.map_idx)
    for t, n in zip(*np.nonzero(done)):
        curriculum.record_episode(bool(success[t, n]), int(steps[t, n]), bank.paths[int(map_idx[t, n])])
        if curriculum.level_key() != level:
            break  # the remaining episodes were played on the previous level


def main():
    args = parse_args()

    # Backends initialize lazily, so the platform can still be chosen here
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif args.device in ("gpu", "cuda"):
        jax.config.update("jax_platforms", "cuda")

    print("=" * 65)
    print("        DECO-SWARM: PHASE 1 NAVIGATION TRAINING (JAX)        ")
    print("=" * 65)
    print(f"Device:           {jax.default_backend()} ({jax.devices()[0].device_kind})")
    print(f"Total Timesteps:  {args.total_timesteps:,}")
    print(f"Parallel Envs:    {args.num_envs}")
    print(f"Rollout Steps:    {args.rollout_steps} per env ({args.rollout_steps * args.num_envs:,} per update)")
    print(f"Batch Seq Len:    {args.batch_seq_len}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Entropy Coef:     {args.ent_coef}")
    print(f"Seed:             {args.seed}")
    print(f"Checkpoints:      {args.save_dir}/")
    if args.resume:
        print(f"Resuming From:    {args.resume}")
    if args.tier >= 0:
        print(f"Enforced Tier:    Tier {args.tier} {'(LOCKED)' if args.lock_tier else '(Auto-promote enabled)'}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Curriculum, maps and environment
    configs = load_configs()
    curriculum = MapCurriculum(sample_mode="random")
    bank = MapBank(curriculum.all_map_paths())
    env = NavEnv(configs, bank)

    # 2. Model and trainer
    model = RecurrentActorCritic.from_agent_config(configs.agent)
    ppo_cfg = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        batch_seq_len=args.batch_seq_len,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        lr=args.lr,
    )
    trainer = PPOTrainer(env, model, ppo_cfg)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = trainer.init_params(init_key)

    if args.resume and os.path.exists(args.resume):
        print(f"\n[Resume] Loading weights from {args.resume}...")
        ckpt = load_checkpoint(args.resume)
        params = ckpt["params"]
        if "tier_idx" in ckpt["meta"]:
            curriculum.current_tier_idx = int(ckpt["meta"]["tier_idx"])
        print("[Resume] Weights restored successfully.\n")

    # Override tier if user explicitly specified --tier
    if args.tier >= 0:
        curriculum.set_tier(args.tier, locked=args.lock_tier)
        print(f"[Tier Enforced] Active Tier set to: {curriculum.current_tier.name} (Locked: {args.lock_tier})\n")

    def make_pool() -> MapPool:
        return MapPool(*map(jnp.asarray, bank.pool(curriculum.active_pool_paths())))

    pool = make_pool()
    level = curriculum.level_key()
    key, run_key = jax.random.split(key)
    runner = trainer.init_runner(run_key, params, pool)

    run_meta = {
        "seed": args.seed,
        "ppo": dataclasses.asdict(ppo_cfg),
        "env": dataclasses.asdict(configs.env),
        "rewards": dataclasses.asdict(configs.rewards),
        "agent": dataclasses.asdict(configs.agent),
    }

    def save(path: str, with_optimizer: bool = True, **meta):
        save_checkpoint(path, runner.params, runner.opt_state if with_optimizer else None, **meta, **run_meta)

    global_step = 0
    steps_per_iter = args.num_envs * args.rollout_steps
    start_time = time.time()
    last_save_step = 0
    iteration = 0

    # Catastrophic failure protection & tier tracking
    tier_peaks: Dict[int, float] = {}
    tier_best_states: Dict[int, Any] = {}

    print(f"\n[Curriculum Starting] Active Tier: {curriculum.current_tier.name} | Map: {curriculum.current_map_name}")
    print("[JIT] The first iteration compiles the rollout and update functions; this can take a minute.\n")

    while global_step < args.total_timesteps:
        iteration += 1
        iter_start = time.time()

        # Collect experience
        runner, rollout, stats = trainer.collect(runner, pool)
        global_step += steps_per_iter
        record_finished_episodes(curriculum, bank, jax.device_get(stats))

        # 1. Promotion checkpoint: saved with the weights that earned it (before this update)
        if curriculum.last_promotion_info is not None:
            info = curriculum.last_promotion_info
            curriculum.last_promotion_info = None
            old_tier = info["old_tier"]

            grad_meta = dict(step=global_step, tier_idx=old_tier, success_rate=info["rate"])
            grad_path = os.path.join(args.save_dir, f"model_graduated_tier_{old_tier}.msgpack")
            save(grad_path, **grad_meta)
            save(os.path.join(args.save_dir, f"model_best_tier_{old_tier}.msgpack"), **grad_meta)
            save(os.path.join(args.save_dir, "model_best.msgpack"), **grad_meta)

            print(
                f"\n[Curriculum Promotion] *** MASTERED {info['old_name']} ({info['rate']:.1%}) "
                f"-> PROMOTED TO {info['new_name']}! ***"
            )
            print(f"  >>> [GRADUATION CHECKPOINT] Saved mastered Tier {old_tier} weights to {grad_path}\n")

        # Update policy
        runner, metrics = trainer.train(runner, rollout)
        metrics = jax.device_get(metrics)

        # Tier (or sequential map) changed: restart all episodes on the new map pool
        if curriculum.level_key() != level:
            pool = make_pool()
            level = curriculum.level_key()
            key, reset_key = jax.random.split(key)
            env_state, obs, hidden = trainer.reset_envs(reset_key, pool)
            runner = runner._replace(env_state=env_state, obs=obs, hidden=hidden)

        iter_time = time.time() - iter_start
        fps = int(steps_per_iter / max(1e-5, iter_time))

        succ_rate = curriculum.success_rate
        avg_steps = curriculum.avg_steps
        curr_tier_id = curriculum.current_tier.tier_id

        # Log progress
        print(
            f"[Step {global_step:6d}/{args.total_timesteps}] "
            f"Tier: {curr_tier_id} ({curriculum.current_map_name}) | "
            f"Success: {succ_rate:5.1%} | "
            f"AvgSteps: {avg_steps:5.1f} | "
            f"PiLoss: {float(metrics['policy_loss']):+6.3f} | "
            f"VLoss: {float(metrics['value_loss']):5.3f} | "
            f"Ent: {float(metrics['entropy']):4.2f} | "
            f"FPS: {fps}"
        )

        # 2. Tier-scoped milestone best tracking (saves only on significant 10% jumps)
        prev_tier_peak = tier_peaks.get(curr_tier_id, 0.0)
        if (
            len(curriculum.recent_successes) >= curriculum.window_size
            and succ_rate >= 0.70
            and succ_rate >= prev_tier_peak + 0.10
        ):
            tier_peaks[curr_tier_id] = succ_rate
            tier_best_states[curr_tier_id] = jax.device_get(runner.params)

            milestone = dict(
                step=global_step,
                tier_idx=curr_tier_id,
                map_idx=curriculum.current_map_idx,
                success_rate=succ_rate,
                avg_steps=avg_steps,
            )
            tier_best_path = os.path.join(args.save_dir, f"model_best_tier_{curr_tier_id}.msgpack")
            save(tier_best_path, **milestone)
            save(os.path.join(args.save_dir, "model_best.msgpack"), **milestone)
            print(f"  [TIER {curr_tier_id} MILESTONE] New high ({succ_rate:.1%})! Saved {tier_best_path}")

        # 3. Intra-tier collapse protection (only within the SAME tier after established proficiency)
        if (
            curr_tier_id in tier_peaks
            and tier_peaks[curr_tier_id] >= 0.70
            and succ_rate <= 0.20
            and curriculum.episodes_in_current_level >= 40
            and curr_tier_id in tier_best_states
        ):
            print(
                f"\n  [ALERT: CATASTROPHIC COLLAPSE IN TIER {curr_tier_id}] "
                f"Success dropped from {tier_peaks[curr_tier_id]:.1%} to {succ_rate:.1%}!"
            )
            print(f"  >>> Rolling back to model_best_tier_{curr_tier_id} and annealing learning rate by 50%...")
            current_lr = get_learning_rate(runner.opt_state) * 0.5
            runner = runner._replace(
                params=jax.device_put(tier_best_states[curr_tier_id]),
                opt_state=set_learning_rate(runner.opt_state, current_lr),
            )
            print(f"  >>> Resumed with annealed LR: {current_lr:.6f}\n")

            curriculum.recent_successes.clear()
            curriculum.recent_steps.clear()

        # 4. Periodic checkpoint saving
        if global_step - last_save_step >= args.save_interval:
            periodic = dict(
                step=global_step,
                tier_idx=curr_tier_id,
                map_idx=curriculum.current_map_idx,
                success_rate=succ_rate,
                avg_steps=avg_steps,
            )
            save_path = os.path.join(args.save_dir, f"model_step_{global_step}.msgpack")
            save(save_path, **periodic)
            save(os.path.join(args.save_dir, "model_latest.msgpack"), **periodic)
            last_save_step = global_step
            print(f"  >>> Periodic checkpoint saved to {save_path} & model_latest.msgpack")

    # Final save
    final_path = os.path.join(args.save_dir, "model_final.msgpack")
    latest_path = os.path.join(args.save_dir, "model_latest.msgpack")
    final_meta = dict(
        step=global_step,
        tier_idx=curriculum.current_tier.tier_id,
        map_idx=curriculum.current_map_idx,
        success_rate=curriculum.success_rate,
        avg_steps=curriculum.avg_steps,
    )
    save(latest_path, with_optimizer=False, **final_meta)
    save(final_path, with_optimizer=False, **final_meta)

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Training completed in {total_time:.1f}s")
    print(f"  - Final Model:    {final_path}")
    print(f"  - Latest Model:   {latest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
