# src/deco_swarm/rendering/camera.py
from typing import Tuple


class Camera:
    """
    Dünya koordinatları (World Space) ile Ekran pikselleri (Screen Space) 
    arasındaki doğrusal dönüşümü yönetir.
    Sorumluluğu: YALNIZCA koordinat ve ölçek dönüşümüdür.
    """

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Kameranın odaklandığı dünya koordinatı (Varsayılan: haritanın merkezi)
        self.x: float = screen_width / 2.0
        self.y: float = screen_height / 2.0
        
        # Zoom katsayısı (1.0 = 1:1 ölçek, 0.5 = 2x uzaklaşma, 2.0 = 2x yakınlaşma)
        self.zoom: float = 1.0
        self.min_zoom: float = 0.2
        self.max_zoom: float = 4.0

    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Dünya koordinatını ekrandaki piksel koordinatına çevirir."""
        wx, wy = world_pos
        sx = (wx - self.x) * self.zoom + (self.screen_width / 2.0)
        sy = (wy - self.y) * self.zoom + (self.screen_height / 2.0)
        return int(round(sx)), int(round(sy))

    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Ekrandaki piksel koordinatını gerçek dünya koordinatına çevirir."""
        sx, sy = screen_pos
        wx = (sx - (self.screen_width / 2.0)) / self.zoom + self.x
        wy = (sy - (self.screen_height / 2.0)) / self.zoom + self.y
        return wx, wy

    def scale_to_screen(self, length: float) -> int:
        """Yarıçap veya çizgi kalınlığı gibi bir uzunluğu zoom oranına göre ölçekler."""
        return max(1, int(round(length * self.zoom)))

    def follow(self, target_pos: Tuple[float, float], smoothing: float = 1.0):
        """
        Kamerayı bir ajanın veya noktanın üzerine odaklar.
        smoothing: 1.0 anında odaklanma, 0.1 yumuşak kamera takibi.
        """
        tx, ty = target_pos
        self.x += (tx - self.x) * smoothing
        self.y += (ty - self.y) * smoothing

    def adjust_zoom(self, factor: float):
        """Zoom oranını sınırlar dahilinde çarparak günceller."""
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))

    def pan(self, dx_screen: float, dy_screen: float):
        """Fareyle ekranı sürüklerken kamerayı kaydırır."""
        self.x -= dx_screen / self.zoom
        self.y -= dy_screen / self.zoom