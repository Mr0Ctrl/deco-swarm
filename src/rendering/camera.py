"""Virtual 2D camera with world-to-screen transforms, pan, zoom, and smooth follow."""

from typing import Sequence


class Camera:
    """2D viewport camera mapping world coordinates to screen pixels."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        x: float = 0.0,
        y: float = 0.0,
        zoom: float = 0.75,
        min_zoom: float = 0.2,
        max_zoom: float = 4.0,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = float(x)
        self.y = float(y)
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self.zoom = max(self.min_zoom, min(self.max_zoom, float(zoom)))

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates (wx, wy) to integer screen pixel coordinates."""
        sx = (wx - self.x) * self.zoom + self.screen_width * 0.5
        sy = (wy - self.y) * self.zoom + self.screen_height * 0.5
        return round(sx), round(sy)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert screen pixel coordinates (sx, sy) back to world coordinates."""
        wx = (sx - self.screen_width * 0.5) / self.zoom + self.x
        wy = (sy - self.screen_height * 0.5) / self.zoom + self.y
        return wx, wy

    def scale_to_screen(self, length: float) -> int:
        """Scale a world-space length into screen pixels (at least 1 px)."""
        return max(1, round(length * self.zoom))

    def adjust_zoom(self, factor: float) -> None:
        """Multiply zoom by factor, clamped to [min_zoom, max_zoom]."""
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))

    def pan(self, dx_screen: float, dy_screen: float) -> None:
        """Pan camera center by screen drag delta."""
        self.x -= dx_screen / self.zoom
        self.y -= dy_screen / self.zoom

    def follow(self, target_pos: Sequence[float], smoothing: float = 0.1) -> None:
        """Smoothly interpolate camera center toward target_pos."""
        self.x += (float(target_pos[0]) - self.x) * smoothing
        self.y += (float(target_pos[1]) - self.y) * smoothing
