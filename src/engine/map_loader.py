# src/deco_swarm/engine/map_loader.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml
from src.engine.map import Segment


@dataclass
class MapConfig:
    name: str
    width: float
    height: float
    segments: List[Segment]
    spawns: Dict[str, Tuple[float, float, float]]  # agent_id -> (x, y, angle)
    goal_pos: Tuple[float, float]
    goal_radius: float


class MapLoader:
    @staticmethod
    def load(file_path: str) -> MapConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        w = float(data["dimensions"]["width"])
        h = float(data["dimensions"]["height"])
        segments: List[Segment] = []

        # 1. Dış Çerçeve Sınırlarını Otomatik Ekle
        segments.append(((0.0, 0.0), (w, 0.0)))
        segments.append(((w, 0.0), (w, h)))
        segments.append(((w, h), (0.0, h)))
        segments.append(((0.0, h), (0.0, 0.0)))

        # 2. Özel Çizgi Parçalarını Ekle
        walls = data.get("walls", {})
        for seg in walls.get("segments", []):
            p1 = (float(seg[0][0]), float(seg[0][1]))
            p2 = (float(seg[1][0]), float(seg[1][1]))
            segments.append((p1, p2))

        # 3. Kutuları 4 Çizgi Parçasına Dönüştürerek Ekle
        for box in walls.get("boxes", []):
            bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            tl = (bx, by)
            tr = (bx + bw, by)
            br = (bx + bw, by + bh)
            bl = (bx, by + bh)
            segments.extend([(tl, tr), (tr, br), (br, bl), (bl, tl)])

        # 4. Ajan Doğuş Noktalarını Çözümle
        spawns = {}
        for agent_id, spawn_data in data.get("spawns", {}).items():
            spawns[agent_id] = (float(spawn_data[0]), float(spawn_data[1]), float(spawn_data[2]))

        # 5. Hedef Bilgisini Çözümle
        goal_data = data.get("goal", {})
        goal_pos = (float(goal_data["position"][0]), float(goal_data["position"][1]))
        goal_radius = float(goal_data.get("radius", 30.0))

        return MapConfig(
            name=data.get("name", "Unnamed Map"),
            width=w,
            height=h,
            segments=segments,
            spawns=spawns,
            goal_pos=goal_pos,
            goal_radius=goal_radius,
        )