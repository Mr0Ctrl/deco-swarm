# deco/maps.py
"""
Map loading and batching.

`load_map` parses the same YAML format as the original MapLoader (outer frame + free segments + boxes).
`MapBank` stacks several maps into fixed-shape arrays so a jitted environment can switch maps by index.
Maps with fewer walls are padded with zero-length segments, which the collision and raycast code ignore
(same degenerate-segment rules as the original engine), so padding never changes the physics.
"""
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import yaml

PAD_SEGMENT = (-1.0e4, -1.0e4, -1.0e4, -1.0e4)


@dataclass
class MapData:
    name: str
    path: str
    width: float
    height: float
    segments: np.ndarray                      # (S, 4) [x1, y1, x2, y2]
    spawn: Tuple[float, float, float]          # agent_0 spawn (x, y, angle_deg)
    goal_pos: Tuple[float, float]
    goal_radius: float

    @property
    def segment_list(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return [((s[0], s[1]), (s[2], s[3])) for s in self.segments.tolist()]


def load_map(file_path: str) -> MapData:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    w = float(data["dimensions"]["width"])
    h = float(data["dimensions"]["height"])

    # 1. Outer frame
    segments = [
        (0.0, 0.0, w, 0.0),
        (w, 0.0, w, h),
        (w, h, 0.0, h),
        (0.0, h, 0.0, 0.0),
    ]

    # 2. Free wall segments
    walls = data.get("walls") or {}
    for seg in walls.get("segments") or []:
        segments.append((float(seg[0][0]), float(seg[0][1]), float(seg[1][0]), float(seg[1][1])))

    # 3. Boxes
    for box in walls.get("boxes") or []:
        bx, by, bw, bh = (float(v) for v in box[:4])
        segments.extend([
            (bx, by, bx + bw, by),
            (bx + bw, by, bx + bw, by + bh),
            (bx + bw, by + bh, bx, by + bh),
            (bx, by + bh, bx, by),
        ])

    spawn = data.get("spawns", {}).get("agent_0", [0.0, 0.0, 0.0])
    goal = data.get("goal", {})

    return MapData(
        name=data.get("name", "Unnamed"),
        path=file_path,
        width=w,
        height=h,
        segments=np.asarray(segments, dtype=np.float64),
        spawn=(float(spawn[0]), float(spawn[1]), float(spawn[2])),
        goal_pos=(float(goal["position"][0]), float(goal["position"][1])),
        goal_radius=float(goal.get("radius", 30.0)),
    )


class MapBank:
    """A fixed set of maps stacked into arrays: segments (M, S_max, 4), spawn (M, 3), goal (M, 2), ..."""

    def __init__(self, paths: Sequence[str]):
        self.paths: List[str] = [os.path.normpath(p) for p in paths]
        self.maps: List[MapData] = [load_map(p) for p in paths]

        max_segments = max(len(m.segments) for m in self.maps)
        segs = np.tile(np.asarray(PAD_SEGMENT, dtype=np.float64), (len(self.maps), max_segments, 1))
        for i, m in enumerate(self.maps):
            segs[i, : len(m.segments)] = m.segments

        self.segments = segs
        self.spawn = np.asarray([m.spawn for m in self.maps], dtype=np.float64)
        self.goal_pos = np.asarray([m.goal_pos for m in self.maps], dtype=np.float64)
        self.goal_radius = np.asarray([m.goal_radius for m in self.maps], dtype=np.float64)
        self.max_map_distance = np.asarray([np.hypot(m.width, m.height) for m in self.maps], dtype=np.float64)
        self.max_width = max(m.width for m in self.maps)
        self.max_height = max(m.height for m in self.maps)

    def __len__(self) -> int:
        return len(self.maps)

    def index_of(self, path: str) -> int:
        return self.paths.index(os.path.normpath(path))

    def pool(self, paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes a set of maps as a fixed-size index array + count, so changing it never recompiles."""
        idx = [self.index_of(p) for p in paths]
        padded = np.zeros(len(self), dtype=np.int32)
        padded[: len(idx)] = idx
        return padded, np.asarray(len(idx), dtype=np.int32)
