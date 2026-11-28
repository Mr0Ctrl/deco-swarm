"""Read-only 5-layer Pygame renderer for 2D Capture the Flag."""

import math
from typing import Sequence

import numpy as np
import pygame

from src.configs import AgentConfig, ArenaData
from src.sim import CTFState

from .camera import Camera

BG_COLOR = (24, 28, 36)
WALL_COLOR = (180, 190, 205)
RED_SOLID = (231, 76, 60)
BLUE_SOLID = (52, 152, 219)
RED_FILL = (231, 76, 60, 70)
BLUE_FILL = (52, 152, 219, 70)
HEADING_COLOR = (241, 196, 15)
CARRIER_RING = (241, 196, 15)
BEAM_COLOR = (120, 240, 255)
RAY_WALL = (231, 76, 60, 55)
RAY_ALLY = (46, 204, 113, 110)
RAY_ENEMY = (243, 156, 18, 130)
HUD_TEXT = (235, 240, 245)


class PygameRenderer:
    """Strictly read-only 5-layer 2D CTF viewer."""

    def __init__(self, width: int, height: int, title: str = "DECO-SWARM v2") -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("monospace", 15)
        self.score_font = pygame.font.SysFont("monospace", 22, bold=True)

    def _draw_zones_and_flags(
        self,
        camera: Camera,
        arena: ArenaData,
        state: CTFState,
    ) -> None:
        """Layer 1: Translucent Red/Blue bases and flags."""
        red_base = np.asarray(arena.red_base)
        blue_base = np.asarray(arena.blue_base)
        flag_r = float(arena.flag_radius)

        for base_arr, fill_col, border_col in (
            (red_base, RED_FILL, RED_SOLID),
            (blue_base, BLUE_FILL, BLUE_SOLID),
        ):
            center = camera.world_to_screen(float(base_arr[0]), float(base_arr[1]))
            rad = camera.scale_to_screen(float(base_arr[2]))
            pygame.draw.circle(self.overlay, fill_col, center, rad)
            pygame.draw.circle(self.screen, border_col, center, rad, 2)

        for flag_pos, is_home, col in (
            (np.asarray(state.red_flag_pos), bool(state.red_flag_home), RED_SOLID),
            (np.asarray(state.blue_flag_pos), bool(state.blue_flag_home), BLUE_SOLID),
        ):
            fc = camera.world_to_screen(float(flag_pos[0]), float(flag_pos[1]))
            fr = camera.scale_to_screen(flag_r)
            pygame.draw.circle(self.screen, col, fc, fr, 2 if is_home else 3)
            pygame.draw.circle(self.screen, (255, 255, 255), fc, max(2, fr // 3))

    def _draw_walls(self, camera: Camera, arena: ArenaData) -> None:
        """Layer 2: Active arena walls and obstacle line segments."""
        segs = np.asarray(arena.segments)
        mask = np.asarray(arena.segment_mask)
        for seg, active in zip(segs, mask):
            if not active:
                continue
            p1 = camera.world_to_screen(float(seg[0, 0]), float(seg[0, 1]))
            p2 = camera.world_to_screen(float(seg[1, 0]), float(seg[1, 1]))
            pygame.draw.line(self.screen, WALL_COLOR, p1, p2, 2)

    def _draw_rays(
        self,
        camera: Camera,
        state: CTFState,
        rays: np.ndarray,
        agent_idx: int,
        agent_cfg: AgentConfig,
    ) -> None:
        """Layer 3: Toggleable panoramic semantic rays for watched agent."""
        if not bool(state.alive[agent_idx]):
            return
        pos = np.asarray(state.pos[agent_idx])
        heading_rad = math.radians(float(state.heading_deg[agent_idx]))
        origin_s = camera.world_to_screen(float(pos[0]), float(pos[1]))
        agent_rays = rays[agent_idx]  # (R, 12)

        for r in agent_rays:
            dist = float(r[0]) * agent_cfg.max_distance
            cos_r, sin_r = float(r[10]), float(r[11])
            # Rotate local ray direction by agent heading
            dx = cos_r * math.cos(heading_rad) - sin_r * math.sin(heading_rad)
            dy = cos_r * math.sin(heading_rad) + sin_r * math.cos(heading_rad)
            end_w = (float(pos[0]) + dx * dist, float(pos[1]) + dy * dist)
            end_s = camera.world_to_screen(end_w[0], end_w[1])

            if r[3] > 0.5:
                col = RAY_ENEMY
            elif r[2] > 0.5:
                col = RAY_ALLY
            else:
                col = RAY_WALL
            pygame.draw.line(self.overlay, col, origin_s, end_s, 1)

            if r[1] > 0.5 or r[2] > 0.5 or r[3] > 0.5:
                pygame.draw.circle(self.screen, col[:3], end_s, 2)

    def _draw_agents_and_beams(
        self,
        camera: Camera,
        state: CTFState,
        agent_cfg: AgentConfig,
        watched_idx: int,
    ) -> None:
        """Layer 4: Agents, team colors, headings, flag-carrier rings, and laser beams."""
        positions = np.asarray(state.pos)
        headings = np.asarray(state.heading_deg)
        teams = np.asarray(state.teams)
        alive = np.asarray(state.alive)
        carrying = np.asarray(state.carrying_flag)
        beam_fired = np.asarray(state.last_beam_fired)
        beam_end = np.asarray(state.last_beam_end)

        rad_s = camera.scale_to_screen(agent_cfg.radius)

        for i in range(len(positions)):
            col = RED_SOLID if int(teams[i]) == 0 else BLUE_SOLID
            center = camera.world_to_screen(float(positions[i, 0]), float(positions[i, 1]))

            if not bool(alive[i]):
                pygame.draw.circle(self.screen, col, center, rad_s, 1)
                continue

            if bool(beam_fired[i]):
                b_end = camera.world_to_screen(float(beam_end[i, 0]), float(beam_end[i, 1]))
                pygame.draw.line(self.screen, BEAM_COLOR, center, b_end, 3)

            pygame.draw.circle(self.screen, col, center, rad_s)
            if i == watched_idx:
                pygame.draw.circle(self.screen, (255, 255, 255), center, rad_s + 2, 1)
            if bool(carrying[i]):
                pygame.draw.circle(self.screen, CARRIER_RING, center, rad_s + 4, 2)

            h_rad = math.radians(float(headings[i]))
            tip_w = (
                float(positions[i, 0]) + math.cos(h_rad) * (agent_cfg.radius + 8.0),
                float(positions[i, 1]) + math.sin(h_rad) * (agent_cfg.radius + 8.0),
            )
            tip_s = camera.world_to_screen(tip_w[0], tip_w[1])
            pygame.draw.line(self.screen, HEADING_COLOR, center, tip_s, 2)

    def render(
        self,
        camera: Camera,
        arena: ArenaData,
        state: CTFState,
        agent_cfg: AgentConfig,
        rays: np.ndarray | None = None,
        watched_idx: int = 0,
        show_rays: bool = True,
        fps: float = 0.0,
        hud_lines: Sequence[str] = (),
    ) -> None:
        """Render one complete read-only frame in 5 ordered layers."""
        self.screen.fill(BG_COLOR)
        self.overlay.fill((0, 0, 0, 0))

        self._draw_zones_and_flags(camera, arena, state)
        self._draw_walls(camera, arena)
        if show_rays and rays is not None:
            self._draw_rays(camera, state, rays, watched_idx, agent_cfg)
        self._draw_agents_and_beams(camera, state, agent_cfg, watched_idx)

        self.screen.blit(self.overlay, (0, 0))

        # Top-center scoreboard
        scores = np.asarray(state.scores)
        score_surf = self.score_font.render(f"RED {int(scores[0])}  -  {int(scores[1])} BLUE", True, HUD_TEXT)
        self.screen.blit(score_surf, (self.width // 2 - score_surf.get_width() // 2, 10))

        # Top-left telemetry
        lines = [f"FPS: {fps:.1f} | Zoom: {camera.zoom:.2f}x", *hud_lines]
        for idx, text in enumerate(lines):
            surf = self.font.render(text, True, HUD_TEXT)
            self.screen.blit(surf, (12, 12 + idx * 18))

        pygame.display.flip()

    def close(self) -> None:
        """Shut down Pygame cleanly."""
        pygame.quit()
