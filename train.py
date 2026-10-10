# train.py
import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.abspath("src"))

from envs.curriculum import MapCurriculum
from envs.single_agent_env import SingleAgentMazeEnv
from models.actor_critic import RecurrentActorCritic
from training.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DECO-SWARM: Phase 1 Navigation Training")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total environment steps to train")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per rollout buffer collection")
    parser.add_argument("--batch-seq-len", type=int, default=32, help="Sequence length for recurrent unrolling")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient for exploration")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--save-interval", type=int, default=10000, help="Timesteps between saving checkpoints")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--tier", type=int, default=-1, help="Force curriculum tier (0 for faz_1v0.*, 1 for faz_1v1.*, etc.)")
    parser.add_argument("--lock-tier", action="store_true", help="Lock the chosen tier (prevent promotion/demotion)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 65)
    print("           DECO-SWARM: PHASE 1 NAVIGATION TRAINING            ")
    print("=" * 65)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
    print(f"Total Timesteps:  {args.total_timesteps:,}")
    print(f"Rollout Steps:    {args.rollout_steps}")
    print(f"Batch Seq Len:    {args.batch_seq_len}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Entropy Coef:     {args.ent_coef}")
    print(f"Checkpoints:      {args.save_dir}/")
    if args.resume:
        print(f"Resuming From:    {args.resume}")
    if args.tier >= 0:
        print(f"Enforced Tier:    Tier {args.tier} {'(LOCKED)' if args.lock_tier else '(Auto-promote enabled)'}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Initialize Sub-Curriculum and Environment
    curriculum = MapCurriculum(sample_mode="random")
    env = SingleAgentMazeEnv(curriculum=curriculum)

    # 2. Initialize Model
    model = RecurrentActorCritic().to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"\n[Resume] Loading weights from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "tier_idx" in checkpoint:
                curriculum.current_tier_idx = checkpoint["tier_idx"]
        else:
            model.load_state_dict(checkpoint)
        print("[Resume] Weights restored successfully.\n")

    # Override tier if user explicitly specified --tier
    if args.tier >= 0:
        curriculum.set_tier(args.tier, locked=args.lock_tier)
        print(f"[Tier Enforced] Active Tier set to: {curriculum.current_tier.name} (Locked: {args.lock_tier})\n")

    # 3. Initialize Trainer
    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        batch_seq_len=args.batch_seq_len,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
    )

    # 4. Prepare initial state & Best Model Tracking
    obs_dict, _ = env.reset()
    current_obs = torch.tensor(obs_dict["agent_0"], dtype=torch.float32, device=device)
    current_hidden = model.initial_hidden_state(batch_size=1, device=device)

    start_time = time.time()
    last_save_step = 0
    iteration = 0

    # Catastrophic failure protection trackers
    best_success_rate = 0.0
    best_avg_steps = 9999.0
    best_model_state = None
    min_episodes_for_peak = 15

    print(f"\n[Curriculum Starting] Active Tier: {curriculum.current_tier.name} | Map: {curriculum.current_map_name}\n")

    while trainer.global_step < args.total_timesteps:
        iteration += 1
        iter_start = time.time()

        # Collect experience
        current_obs, current_hidden = trainer.collect_rollouts(current_obs, current_hidden)

        # Update policy
        train_metrics = trainer.train_epoch()

        iter_time = time.time() - iter_start
        fps = int(args.rollout_steps / max(1e-5, iter_time))

        succ_rate = curriculum.success_rate
        avg_steps = curriculum.avg_steps

        # Log progress
        print(
            f"[Step {trainer.global_step:6d}/{args.total_timesteps}] "
            f"Tier: {curriculum.current_tier.tier_id} ({curriculum.current_map_name}) | "
            f"Success: {succ_rate:5.1%} | "
            f"AvgSteps: {avg_steps:5.1f} | "
            f"PiLoss: {train_metrics['policy_loss']:+6.3f} | "
            f"VLoss: {train_metrics['value_loss']:5.3f} | "
            f"Ent: {train_metrics['entropy']:4.2f} | "
            f"FPS: {fps}"
        )

        # 1. Best Model Tracking (Protection Rule A: Save peak performance)
        is_new_best = False
        if len(curriculum.recent_successes) >= min_episodes_for_peak:
            if succ_rate > best_success_rate or (succ_rate == best_success_rate and avg_steps < best_avg_steps and succ_rate > 0.0):
                best_success_rate = succ_rate
                best_avg_steps = avg_steps
                is_new_best = True
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                best_path = os.path.join(args.save_dir, "model_best.pt")
                tier_best_path = os.path.join(args.save_dir, f"model_best_tier_{curriculum.current_tier.tier_id}.pt")

                checkpoint_best = {
                    "step": trainer.global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "tier_idx": curriculum.current_tier_idx,
                    "map_idx": curriculum.current_map_idx,
                    "success_rate": best_success_rate,
                    "avg_steps": best_avg_steps,
                }
                torch.save(checkpoint_best, best_path)
                torch.save(checkpoint_best, tier_best_path)
                print(f"  [BEST MODEL] New peak ({best_success_rate:.1%})! Saved to {best_path}")

        # 2. Catastrophic Collapse Detection & Rollback (Protection Rule B)
        if best_success_rate >= 0.50 and succ_rate <= 0.15 and best_model_state is not None:
            print(
                f"\n  [ALERT: CATASTROPHIC COLLAPSE DETECTED] Success dropped from {best_success_rate:.1%} to {succ_rate:.1%}!"
            )
            print("  >>> Rolling back model to peak weights (model_best.pt) and annealing learning rate by 50%...")
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

            # Decay learning rate to prevent jumping off loss cliffs again
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] *= 0.5
                current_lr = param_group["lr"]
            print(f"  >>> Resumed with annealed LR: {current_lr:.6f}\n")

            # Reset degraded statistics to allow healthy recovery
            curriculum.recent_successes.clear()
            curriculum.recent_steps.clear()

        # 3. Periodic Checkpoint Saving
        if trainer.global_step - last_save_step >= args.save_interval:
            save_path = os.path.join(args.save_dir, f"model_step_{trainer.global_step}.pt")
            latest_path = os.path.join(args.save_dir, "model_latest.pt")
            checkpoint = {
                "step": trainer.global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "tier_idx": curriculum.current_tier_idx,
                "map_idx": curriculum.current_map_idx,
                "success_rate": succ_rate,
                "avg_steps": avg_steps,
            }
            torch.save(checkpoint, save_path)
            torch.save(checkpoint, latest_path)
            last_save_step = trainer.global_step
            print(f"  >>> Checkpoint saved to {save_path} & {latest_path}")

    # Final save (Protection Rule C: Guarantee model_final.pt is safe and healthy)
    final_path = os.path.join(args.save_dir, "model_final.pt")
    latest_path = os.path.join(args.save_dir, "model_latest.pt")

    # Save the absolute latest state
    checkpoint_latest = {
        "step": trainer.global_step,
        "model_state_dict": model.state_dict(),
        "tier_idx": curriculum.current_tier_idx,
        "map_idx": curriculum.current_map_idx,
        "success_rate": curriculum.success_rate,
        "avg_steps": curriculum.avg_steps,
    }
    torch.save(checkpoint_latest, latest_path)

    # If the latest step severely degraded from peak, preserve best model as model_final.pt
    if best_model_state is not None and (curriculum.success_rate < best_success_rate - 0.25):
        print("\n  [SAFE FINAL RULE] Final step policy is degraded compared to peak!")
        print(f"  >>> Protecting model_final.pt: Saving peak weights ({best_success_rate:.1%}) as model_final.pt.")
        checkpoint_final = {
            "step": trainer.global_step,
            "model_state_dict": {k: v.to(device) for k, v in best_model_state.items()},
            "tier_idx": curriculum.current_tier_idx,
            "map_idx": curriculum.current_map_idx,
            "success_rate": best_success_rate,
            "avg_steps": best_avg_steps,
        }
        torch.save(checkpoint_final, final_path)
    else:
        torch.save(checkpoint_latest, final_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Training completed in {total_time:.1f}s")
    print(f"  - Peak Model:     {os.path.join(args.save_dir, 'model_best.pt')}")
    print(f"  - Final Model:    {final_path}")
    print(f"  - Latest Model:   {latest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
