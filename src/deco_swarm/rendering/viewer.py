import jax
import numpy as np
import pygame

from deco_swarm.sim.raycast import ALLY, ENEMY, ENEMY_BASE, ENEMY_FLAG, NONE, OWN_BASE, OWN_FLAG, WALL, cast_rays
from deco_swarm.sim.state import CARRIED, STRAY, carrying_mask

BG = (25, 25, 30)
WALL_COLOR = (200, 200, 200)
TEAM_COLORS = [(220, 70, 70), (70, 120, 235)]        # red, blue
BASE_COLORS = [(90, 35, 35), (35, 50, 95)]
RAY_COLORS = {
    WALL: (120, 120, 120), ALLY: (80, 220, 80), ENEMY: (255, 150, 0),
    OWN_FLAG: (255, 255, 0), ENEMY_FLAG: (255, 0, 255),
    OWN_BASE: (0, 200, 200), ENEMY_BASE: (200, 0, 200),
}


class Viewer:
    """Draws one CTF env state. Meters -> pixels happens only here."""

    def __init__(self, env, render_cfg):
        pygame.init()
        sc, cam = render_cfg.screen, render_cfg.camera
        self.w, self.h = sc.width_px, sc.height_px
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(sc.title)
        self.font = pygame.font.SysFont("monospace", 16)
        self.env = env
        self.ppm = cam.pixels_per_meter
        self.zoom, self.min_zoom, self.max_zoom = cam.default_zoom, cam.min_zoom, cam.max_zoom
        self.cx, self.cy = (float(v) / 2 for v in env.arena.size)        # camera at the map center
        self.arena = jax.device_get(env.arena)
        self._rays = jax.jit(self._cast)

    # ---- world (m, +y up) -> screen (px, +y down) ----
    def to_screen(self, p):
        s = self.ppm * self.zoom
        return int(self.w / 2 + (p[0] - self.cx) * s), int(self.h / 2 - (p[1] - self.cy) * s)

    def px(self, length_m):
        return max(1, int(length_m * self.ppm * self.zoom))

    def zoom_by(self, factor):
        self.zoom = min(self.max_zoom, max(self.min_zoom, self.zoom * factor))

    def _cast(self, state):
        env = self.env
        carrying = carrying_mask(state.flag_state, state.flag_carrier, env.num_agents)
        return cast_rays(state.pos, state.heading_deg, state.alive, carrying, env.team, state.flag_pos,
                         state.flag_state != CARRIED, env.arena, env.cfg.agent.body.radius_m,
                         env.cfg.agent.perception)

    # ---- drawing ----
    def draw(self, state, selected=0, show_rays=True):
        a, env = self.arena, self.env
        s = jax.device_get(state)
        scr = self.screen
        scr.fill(BG)

        for t in (0, 1):                                                  # bases
            pygame.draw.circle(scr, BASE_COLORS[t], self.to_screen(a.base[t, :2]), self.px(a.base[t, 2]))

        for seg, on in zip(a.segments, a.segment_mask):                  # walls
            if on:
                pygame.draw.line(scr, WALL_COLOR, self.to_screen(seg[:2]), self.to_screen(seg[2:]), 2)

        if show_rays:                                                     # rays of the selected agent
            dist, cls, _ = jax.device_get(self._rays(state))
            p = env.cfg.agent.perception
            rel = np.linspace(-p.fov_deg / 2, p.fov_deg / 2, p.ray_count)
            ang = np.deg2rad(s.heading_deg[selected] + rel)
            o = s.pos[selected]
            for d, c, th in zip(dist[selected], cls[selected], ang):
                if c == NONE:
                    continue
                end = o + d * np.array([np.cos(th), np.sin(th)])
                pygame.draw.line(scr, RAY_COLORS[int(c)], self.to_screen(o), self.to_screen(end), 1)

        carrying = jax.device_get(carrying_mask(state.flag_state, state.flag_carrier, env.num_agents))
        for i in range(env.num_agents):                                   # agents
            if not s.alive[i]:
                continue
            t = int(env.team[i])
            c = self.to_screen(s.pos[i])
            r = self.px(env.cfg.agent.body.radius_m)
            pygame.draw.circle(scr, TEAM_COLORS[t], c, r)
            th = np.deg2rad(s.heading_deg[i])
            tip = s.pos[i] + 1.2 * env.cfg.agent.body.radius_m * np.array([np.cos(th), np.sin(th)])
            pygame.draw.line(scr, (255, 255, 255), c, self.to_screen(tip), 2)
            if carrying[i]:
                pygame.draw.circle(scr, (255, 255, 0), c, r + 3, 2)
            if i == selected:
                pygame.draw.circle(scr, (255, 255, 255), c, r + 6, 1)

        for t in (0, 1):                                                  # flags (carried ones ride on the carrier)
            if s.flag_state[t] == CARRIED:
                continue
            c = self.to_screen(s.flag_pos[t])
            pygame.draw.circle(scr, TEAM_COLORS[t], c, self.px(a.flag_radius))
            if s.flag_state[t] == STRAY:
                pygame.draw.circle(scr, (255, 255, 255), c, self.px(a.flag_radius) + 2, 1)

        ep = env.cfg.environment.episode
        secs_left = (ep.max_steps - int(s.step)) * ep.action_repeat / env.cfg.simulation.physics.tick_hz
        hud = (f"RED {int(s.score[0])} : {int(s.score[1])} BLUE   time {secs_left:5.1f}s   "
               f"agent {env.agents[selected]}   [WASD move  <-/-> turn  TAB agent  R rays  wheel zoom]")
        scr.blit(self.font.render(hud, True, (230, 230, 230)), (10, 8))
        pygame.display.flip()

    def close(self):
        pygame.quit()