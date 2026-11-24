# src/engine/map_loader.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml
from engine.core.geometry import Segment2D, Point2D
from engine.entities.map import GameMap


@dataclass
class MapConfig:
    name: str
    width: float
    height: float
    segments: List[Segment2D]
    spawns: Dict[str, Tuple[float, float, float]]  # agent_id -> (x, y, angle)
    goal_pos: Point2D
    goal_radius: float


class MapLoader:
    @staticmethod
    def load(file_path: str) -> MapConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        w = float(data["dimensions"]["width"])
        h = float(data["dimensions"]["height"])
        segments: List[Segment2D] = []

        # 1. Dış Çerçeve
        segments.append(((0.0, 0.0), (w, 0.0)))
        segments.append(((w, 0.0), (w, h)))
        segments.append(((w, h), (0.0, h)))
        segments.append(((0.0, h), (0.0, 0.0)))

        # 2. Duvarlar
        walls = data.get("walls", {})
        for seg in walls.get("segments", []):
            p1 = (float(seg[0][0]), float(seg[0][1]))
            p2 = (float(seg[1][0]), float(seg[1][1]))
            segments.append((p1, p2))

        # 3. Kutular
        for box in walls.get("boxes", []):
            bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            segments.extend([
                ((bx, by), (bx + bw, by)),
                ((bx + bw, by), (bx + bw, by + bh)),
                ((bx + bw, by + bh), (bx, by + bh)),
                ((bx, by + bh), (bx, by)),
            ])

        spawns = {
            k: (float(v[0]), float(v[1]), float(v[2]))
            for k, v in data.get("spawns", {}).items()
        }

        goal = data.get("goal", {})
        goal_pos = (float(goal["position"][0]), float(goal["position"][1]))
        goal_radius = float(goal.get("radius", 30.0))

        return MapConfig(
            name=data.get("name", "Unnamed"),
            width=w,
            height=h,
            segments=segments,
            spawns=spawns,
            goal_pos=goal_pos,
            goal_radius=goal_radius,
        )