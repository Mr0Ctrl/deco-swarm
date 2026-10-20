# src/envs/vec_env.py
import numpy as np
from typing import List, Dict, Any, Tuple
from envs.single_agent_env import SingleAgentMazeEnv
from envs.curriculum import MapCurriculum


class VectorMazeEnv:
    """
    Synchronous Vectorized Environment for DECO-SWARM.
    Manages N independent SingleAgentMazeEnv instances concurrently.
    Each sub-environment can run on a different map sampled from the curriculum,
    providing diverse spatial experience in parallel with batched GPU inference.
    """

    def __init__(self, num_envs: int, curriculum: MapCurriculum, **env_kwargs):
        self.num_envs = num_envs
        self.curriculum = curriculum

        # Instantiate N independent environments sharing the same curriculum manager
        self.envs = [
            SingleAgentMazeEnv(curriculum=curriculum, **env_kwargs)
            for _ in range(num_envs)
        ]

        self.action_repeat = self.envs[0].action_repeat
        self.max_steps = self.envs[0].max_steps

    def reset(self) -> np.ndarray:
        """
        Resets all N environments.
        Returns:
            batched_obs: (num_envs, 640) float32 numpy array
        """
        obs_list = []
        for env in self.envs:
            obs_dict, _ = env.reset()
            obs_list.append(obs_dict["agent_0"])
        return np.stack(obs_list, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Steps all N environments synchronously.
        Args:
            actions: (num_envs, 2) numpy array of actions [move_idx, rot_idx]
        Returns:
            next_obs: (num_envs, 640) float32 numpy array
            rewards: (num_envs,) float32 numpy array
            dones: (num_envs,) bool numpy array
            infos: List of dicts per environment
        """
        next_obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for i, env in enumerate(self.envs):
            act = actions[i]
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step({"agent_0": act})

            reward = rew_dict["agent_0"]
            done = term_dict["agent_0"] or trunc_dict["agent_0"]

            if done:
                # Auto-reset completed sub-environment
                reset_obs_dict, _ = env.reset()
                next_obs = reset_obs_dict["agent_0"]
            else:
                next_obs = obs_dict["agent_0"]

            next_obs_list.append(next_obs)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info_dict["agent_0"])

        return (
            np.stack(next_obs_list, axis=0).astype(np.float32),
            np.array(rewards_list, dtype=np.float32),
            np.array(dones_list, dtype=bool),
            infos_list,
        )
