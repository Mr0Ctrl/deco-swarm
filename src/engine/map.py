# src/deco_swarm/engine/map.py
from typing import List, Tuple

# Doğru parçası tipi: ((x1, y1), (x2, y2))
Segment = Tuple[Tuple[float, float], Tuple[float, float]]


class GameMap:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.segments: List[Segment] = []
        self._build_default_arena()

    def _build_default_arena(self):
        """
        Dış çerçeveyi, bir odayı ve kör noktaları test etmek için 
        açılı/düz siperleri haritaya ekler.
        """
        w, h = self.width, self.height

        # 1. Dış Çerçeve (Harita sınırları)
        self.add_segment((0, 0), (w, 0))          # Üst
        self.add_segment((w, 0), (w, h))          # Sağ
        self.add_segment((w, h), (0, h))          # Alt
        self.add_segment((0, h), (0, 0))          # Sol

        # 2. İç Siperler ve Odalar (Taktiksel test için)
        # Ortada dikey bir siper
        self.add_segment((w * 0.5, h * 0.2), (w * 0.5, h * 0.6))
        
        # Açılı bir siper (Raycast açılı yüzeyleri kaçırıyor mu testi için)
        self.add_segment((w * 0.2, h * 0.7), (w * 0.4, h * 0.85))

        # Küçük bir kutu / siper
        self.add_box(x=w * 0.7, y=h * 0.4, box_w=80, box_h=80)

    def add_segment(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.segments.append((p1, p2))

    def add_box(self, x: float, y: float, box_w: float, box_h: float):
        """Bir kutuyu 4 adet çizgi parçası olarak haritaya ekler."""
        top_left = (x, y)
        top_right = (x + box_w, y)
        bottom_right = (x + box_w, y + box_h)
        bottom_left = (x, y + box_h)

        self.add_segment(top_left, top_right)
        self.add_segment(top_right, bottom_right)
        self.add_segment(bottom_right, bottom_left)
        self.add_segment(bottom_left, top_left)