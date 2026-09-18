# watch.py
"""Eğitilmiş bir modeli pencerede izlemek için. train.py'nin ürettiği modeli çalıştırır.

Kullanım:
    python watch.py                  # önce models/best_model.zip, yoksa ppo_deco_swarm.zip
    python watch.py <model_yolu.zip> # belirli bir modeli çalıştır
"""
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from stable_baselines3 import PPO
from src.envs.single_agent_env import DecoSwarmEnv

DEFAULT_BEST = os.path.join("models", "best_model.zip")
DEFAULT_LATEST = "ppo_deco_swarm.zip"


def resolve_model_path(explicit_path: str = None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Model bulunamadı: {explicit_path}")
        return explicit_path

    if os.path.exists(DEFAULT_BEST):
        return DEFAULT_BEST
    if os.path.exists(DEFAULT_LATEST):
        return DEFAULT_LATEST

    raise FileNotFoundError(
        "Eğitilmiş model bulunamadı. Önce 'python train.py' ile bir model eğitin."
    )


def main():
    explicit_path = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = resolve_model_path(explicit_path)
    print(f"Model yükleniyor: {model_path}")

    env = DecoSwarmEnv(render_mode="human")
    model = PPO.load(model_path)

    episode = 0
    obs, info = env.reset()
    try:
        while not env.window_closed():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episode += 1
                status = "HEDEFE ULAŞTI" if terminated else "SÜRE DOLDU"
                print(f"Bölüm {episode}: {status}")
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
