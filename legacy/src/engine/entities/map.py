# src/engine/entities/map.py
from typing import List, Tuple
from engine.core.geometry import Segment2D, Point2D


class GameMap:
    def __init__(self, width: float, height: float, segments: List[Segment2D]):
        self.width = width
        self.height = height
        self.segments = segments

    def add_segment(self, p1: Point2D, p2: Point2D):
        self.segments.append((p1, p2))