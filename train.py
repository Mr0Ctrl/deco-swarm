# train.py
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.envs.single_agent_env import DecoSwarmEnv
from watch import resolve_model_path

DEFAULT_ADDITIONAL_TIMESTEPS = 300_000
FINAL_VISIBLE_STEPS = 500
MODELS_DIR = "models"


def main():
    # Komut satırından kaç adım daha eğitileceği verilebilir: `python train.py 500000`
    additional_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ADDITIONAL_TIMESTEPS

    headless_env = DecoSwarmEnv(render_mode=None)

    # Daha önce eğitilmiş bir model varsa (models/best_model.zip veya ppo_deco_swarm.zip)
    # sıfırdan başlamak yerine onun üzerine eğitime devam edilir.
    try:
        existing_model_path = resolve_model_path()
    except FileNotFoundError:
        existing_model_path = None

    if existing_model_path:
        print(f"Var olan model devam ettiriliyor: {existing_model_path}")
        model = PPO.load(existing_model_path, env=headless_env)
    else:
        print("Var olan model bulunamadı, sıfırdan yeni model oluşturuluyor.")
        model = PPO("MlpPolicy", headless_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]))

    # Periyodik değerlendirme yapıp en iyi performanslı checkpoint'i models/best_model.zip'e kaydeder
    eval_env = Monitor(DecoSwarmEnv(render_mode=None))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=MODELS_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    headless_steps = max(0, additional_timesteps - FINAL_VISIBLE_STEPS)
    if headless_steps > 0:
        model.learn(total_timesteps=headless_steps, callback=eval_callback, reset_num_timesteps=False)
    headless_env.close()
    eval_env.close()

    # 2. Son FINAL_VISIBLE_STEPS adım penceresi açık şekilde çalışır — eğitilmiş modeli izleyebilirsiniz
    visible_env = DecoSwarmEnv(render_mode="human")
    model.set_env(visible_env)
    model.learn(total_timesteps=FINAL_VISIBLE_STEPS, reset_num_timesteps=False)
    visible_env.close()

    model.save("ppo_deco_swarm")


if __name__ == "__main__":
    main()
