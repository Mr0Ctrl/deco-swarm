# src/envs/single_agent_env.py
import math
import os
from collections import deque

import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.engine.map_loader import MapLoader
from src.engine.world import World
from src.engine.entities.agent import ActionDecoder

AGENT_ID = "agent_0"

DIST_REWARD_SCALE = 5.0
GOAL_BONUS = 10.0
COLLISION_PENALTY = 0.05
STEP_PENALTY = 0.001

# Navigasyon mesafe alanı (BFS grid) için hücre boyutu (piksel)
NAV_GRID_CELL_SIZE = 20.0


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _point_segment_distance(point, segment) -> float:
    """Bir noktanın bir doğru parçasına en yakın uzaklığı."""
    (x1, y1), (x2, y2) = segment
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


class DecoSwarmEnv(gym.Env):
    """arena_1v0 haritasında tek ajanın hedefe yürümesini öğrenmesi için ince bir Gymnasium sarmalayıcısı."""

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        config_dir: str = "configs",
        map_path: str = "configs/maps/arena_2v0.yaml",
        render_mode: str = "human",
    ):
        super().__init__()
        self.render_mode = render_mode

        self.render_cfg = _load_yaml(os.path.join(config_dir, "render.yaml"))
        self.sim_cfg = _load_yaml(os.path.join(config_dir, "simulation.yaml"))
        self.agent_cfg = _load_yaml(os.path.join(config_dir, "agent.yaml"))
        self.map_cfg = MapLoader.load(map_path)

        self.world = World(map_config=self.map_cfg, sim_config=self.sim_cfg, agent_config=self.agent_cfg)
        self._build_distance_field()

        self.target_fps = self.render_cfg["screen"]["fps"]
        self.dt = 1.0 / self.target_fps
        self.max_steps = self.sim_cfg.get("episode", {}).get("max_steps", 500)

        obs_dim = self.world.agents[AGENT_ID].get_observation().shape[0]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 3])

        self._step_count = 0
        self._prev_distance = 0.0

        self._renderer = None
        self._camera = None
        self._clock = None
        if self.render_mode == "human":
            self._init_renderer()

    def _init_renderer(self):
        from src.rendering.camera import Camera
        from src.rendering.pygame_renderer import PygameRenderer
        import pygame

        screen_w = self.render_cfg["screen"]["width"]
        screen_h = self.render_cfg["screen"]["height"]
        self._renderer = PygameRenderer(screen_w, screen_h)
        self._camera = Camera(screen_w, screen_h)
        self._camera.x = self.map_cfg.width / 2.0
        self._camera.y = self.map_cfg.height / 2.0
        self._camera.zoom = self.render_cfg["camera"]["default_zoom"]
        self._clock = pygame.time.Clock()

    def _build_distance_field(self):
        """
        Harita statik olduğu için bir kere: hücrelere bölünmüş grid üzerinde
        hedeften başlayan BFS ile her hücrenin "engelleri dolaşarak" hedefe
        olan gerçek adım mesafesini hesaplar. Ödülde düz-çizgi mesafesi yerine
        bu kullanılır; aksi halde iki kutunun köşe köşeye değdiği noktada
        (bkz. arena_1v0.yaml) düz-çizgi mesafesi dolaşmayı ödüllendirmiyor ve
        ajan tam orada sıkışıp kalıyordu.
        """
        cell = NAV_GRID_CELL_SIZE
        cols = max(1, int(self.map_cfg.width // cell))
        rows = max(1, int(self.map_cfg.height // cell))
        clearance = self.agent_cfg["body"]["radius"] + 2.0
        segments = self.world.map.segments

        free = [[True] * rows for _ in range(cols)]
        for i in range(cols):
            cx = (i + 0.5) * cell
            for j in range(rows):
                cy = (j + 0.5) * cell
                for seg in segments:
                    if _point_segment_distance((cx, cy), seg) < clearance:
                        free[i][j] = False
                        break

        gx, gy = self.world.goal_pos
        gi = min(cols - 1, max(0, int(gx // cell)))
        gj = min(rows - 1, max(0, int(gy // cell)))
        free[gi][gj] = True  # hedef hücresi her zaman ulaşılabilir sayılır

        dist = [[-1] * rows for _ in range(cols)]
        dist[gi][gj] = 0
        queue = deque([(gi, gj)])
        while queue:
            ci, cj = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < cols and 0 <= nj < rows and free[ni][nj] and dist[ni][nj] == -1:
                    dist[ni][nj] = dist[ci][cj] + 1
                    queue.append((ni, nj))

        self._grid_cell = cell
        self._grid_cols = cols
        self._grid_rows = rows
        self._grid_dist = dist

    def _navigable_distance_to_goal(self) -> float:
        agent = self.world.agents[AGENT_ID]
        i = min(self._grid_cols - 1, max(0, int(agent.x // self._grid_cell)))
        j = min(self._grid_rows - 1, max(0, int(agent.y // self._grid_cell)))

        steps = self._grid_dist[i][j]
        if steps == -1:
            # Hücre BFS ile ulaşılamamış (örn. duvara çok yakın bir anlık konum) —
            # yakın çevredeki en iyi bilinen hücreye düş.
            best = None
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self._grid_cols and 0 <= nj < self._grid_rows:
                        nd = self._grid_dist[ni][nj]
                        if nd != -1 and (best is None or nd < best):
                            best = nd
            steps = best if best is not None else (self._distance_to_goal() / self._grid_cell)

        return steps * self._grid_cell

    def _distance_to_goal(self) -> float:
        """Düz-çizgi mesafesi; yalnızca grid fallback'inde kullanılır."""
        agent = self.world.agents[AGENT_ID]
        gx, gy = self.world.goal_pos
        return ((agent.x - gx) ** 2 + (agent.y - gy) ** 2) ** 0.5

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self._step_count = 0
        self._prev_distance = self._navigable_distance_to_goal()
        obs = self.world.agents[AGENT_ID].get_observation()
        return obs, {}

    def step(self, action):
        move_idx, rot_idx = int(action[0]), int(action[1])
        movement = ActionDecoder.decode((move_idx, rot_idx))

        self.world.step(actions={AGENT_ID: movement}, dt=self.dt)
        self._step_count += 1

        distance = self._navigable_distance_to_goal()
        reward = (self._prev_distance - distance) / self.world.max_map_distance * DIST_REWARD_SCALE
        reward -= STEP_PENALTY
        if self.world.last_collisions.get(AGENT_ID, False):
            reward -= COLLISION_PENALTY
        self._prev_distance = distance

        terminated = self.world.is_goal_reached(AGENT_ID)
        if terminated:
            reward += GOAL_BONUS
        truncated = self._step_count >= self.max_steps

        obs = self.world.agents[AGENT_ID].get_observation()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {}

    def _render_frame(self):
        if self._renderer is None:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._clock.tick(self.target_fps)
        agent = self.world.agents[AGENT_ID]
        hud_info = {
            "Polar Mesafe (r)": f"{agent.polar_target[0]:.3f}",
            "Polar Açı (Δθ)": f"{agent.polar_target[1]:+.3f}",
            "Adım": f"{self._step_count}/{self.max_steps}",
        }
        self._renderer.render(self.world, self._camera, current_fps=self._clock.get_fps(), hud_info=hud_info)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def window_closed(self) -> bool:
        """render_mode='human' iken pencere kullanıcı tarafından kapatıldıysa True döner."""
        return self.render_mode == "human" and self._renderer is None
