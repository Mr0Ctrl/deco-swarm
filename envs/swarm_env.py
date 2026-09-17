# src/envs/swarm_env.py
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.engine.map_loader import MapLoader, MapConfig
from src.engine.world import World
from src.engine.controllers import ActionDecoder
from src.engine.core.geometry import distance


class SwarmParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "deco_swarm_v0"}

    def __init__(
        self,
        map_config: MapConfig,
        sim_config: dict,
        agent_config: dict,
        action_repeat: int = 6,
        max_steps: int = 500,
    ):
        super().__init__()
        self.map_config = map_config
        self.sim_config = sim_config
        self.agent_config = agent_config
        self.action_repeat = action_repeat
        self.max_steps = max_steps

        self.world = World(map_config, sim_config, agent_config)

        # Çoklu Ajan Listesi (Şimdilik 1v0 için sadece 'agent_0')
        self.possible_agents = list(map_config.spawns.keys())
        self.agents = self.possible_agents[:]

        # Fizik adım süresi (60 Hz -> 0.0166 sn)
        self.dt = 1.0 / self.sim_config["physics"].get("tick_rate", 60.0)

        # GÖZLEM UZAYI: 640 Float [-1.0, 1.0]
        # (90 ray x 7 + pusula 2 + hızlar 3 + a_{t-1} 3 + öz durum 2 = 640)
        self.obs_dim = 640
        self._observation_spaces = {
            agent_id: spaces.Box(
                low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
            )
            for agent_id in self.possible_agents
        }

        # AKSİYON UZAYI: Multi-Discrete [Hareket (5), Dönüş (3), Tetik (2)]
        self._action_spaces = {
            agent_id: spaces.MultiDiscrete([5, 3])
            for agent_id in self.possible_agents
        }

        self.step_count = 0
        self.prev_distances: Dict[str, float] = {}

    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.agents = self.possible_agents[:]
        self.world.reset()
        self.step_count = 0

        # Başlangıç mesafelerini kaydet
        self.prev_distances.clear()
        for agent_id, agent in self.world.agents.items():
            self.prev_distances[agent_id] = distance(agent.position, self.world.goal_pos)

        observations = {
            agent_id: self.world.agents[agent_id].get_observation()
            for agent_id in self.agents
        }
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, Any]):
        """
        PettingZoo Standart Step:
        actions: {'agent_0': array([move_idx, rot_idx, fire_idx])}
        """
        self.step_count += 1

        # 1. Ayrık Aksiyonları Fizik Motorunun Float Değerlerine Çözümle
        decoded_actions = {}
        for agent_id, act in actions.items():
            decoded_actions[agent_id] = ActionDecoder.decode(act)

        # 2. ACTION REPEAT (10 Hz Karar -> 60 Hz Fizik Adımı)
        for _ in range(self.action_repeat):
            self.world.step(decoded_actions, dt=self.dt)

        # 3. Gözlemleri Topla, Ödülleri Hesapla
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Zaman aşımı kontrolü (Truncated)
        timed_out = self.step_count >= self.max_steps

        for agent_id in self.agents:
            agent = self.world.agents[agent_id]
            observations[agent_id] = agent.get_observation()

            # --- ÖDÜL MÜHENDİSLİĞİ (REWARD SHAPING) ---
            # A) Hedefe Yaklaşma Ödülü (Delta Distance)
            current_dist = distance(agent.position, self.world.goal_pos)
            prev_dist = self.prev_distances.get(agent_id, current_dist)
            delta_dist = prev_dist - current_dist  # Pozitif = yaklaştı, Negatif = uzaklaştı
            self.prev_distances[agent_id] = current_dist

            # Mesafe ödülünü ölçeklendir (örn: +1 piksel yaklaşma = +0.05 puan)
            step_reward = delta_dist * 0.05

            # B) Zaman Cezası (Ajanı hedefe hızlı gitmeye zorlar)
            step_reward -= 0.01

            # C) Hedefe Ulaşma Başarısı (Zafer!)
            goal_reached = self.world.is_goal_reached(agent_id)
            if goal_reached:
                step_reward += 100.0

            rewards[agent_id] = float(step_reward)
            terminations[agent_id] = goal_reached
            truncations[agent_id] = timed_out
            infos[agent_id] = {"goal_reached": goal_reached}

        # Biten ajanları aktif listeden çıkar
        self.agents = [
            a for a in self.agents if not (terminations[a] or truncations[a])
        ]

        return observations, rewards, terminations, truncations, infos