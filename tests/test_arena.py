import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from deco_swarm.config import ConfigError, compile_arena, load_all, stack_arenas

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PHYS = load_all(CONFIGS).simulation.physics

BASE_MAP = {           # minimal symmetric map; each test changes one thing
    "name": "t", "symmetry": "rotate_180",
    "dimensions": {"width_m": 60.0, "height_m": 40.0},
    "bases": {"red": [6.0, 20.0, 2.5], "blue": [54.0, 20.0, 2.5]},
    "flags": {"red": [6.0, 20.0], "blue": [54.0, 20.0], "radius_m": 0.9},
    "spawns": {"red": [[9.0, 16.0, 0.0], [9.0, 24.0, 0.0]],
               "blue": [[51.0, 24.0, 180.0], [51.0, 16.0, 180.0]]},
    "walls": {"segments": [], "boxes": []},
}


def make(tmp_path, **changes):
    m = copy.deepcopy(BASE_MAP)
    m.update(changes)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(m))
    return p


def test_empty_map_has_border_only(tmp_path):
    a = compile_arena(make(tmp_path), PHYS)
    assert int(a.segment_mask.sum()) == 4


def test_one_box_adds_four(tmp_path):
    walls = {"segments": [], "boxes": [[28.0, 18.0, 4.0, 4.0]]}      # centered -> symmetric
    a = compile_arena(make(tmp_path, walls=walls), PHYS)
    assert int(a.segment_mask.sum()) == 8
    np.testing.assert_allclose(a.segments[4], [28.0, 18.0, 32.0, 18.0])


def test_fixed_shapes(tmp_path):
    a = compile_arena(make(tmp_path), PHYS)
    assert a.segments.shape == (PHYS.max_segments, 4)
    assert a.segment_mask.shape == (PHYS.max_segments,)
    assert a.spawns.shape == (2, PHYS.max_team_size, 3)


def test_too_many_segments(tmp_path):
    segs = [[[30.0, 1.0 + i * 0.1], [30.0, 39.0 - i * 0.1]] for i in range(PHYS.max_segments)]
    with pytest.raises(ConfigError, match="max_segments"):
        compile_arena(make(tmp_path, walls={"segments": segs, "boxes": []}), PHYS)


def test_shifted_spawn_breaks_symmetry(tmp_path):
    spawns = copy.deepcopy(BASE_MAP["spawns"])
    spawns["blue"][0][1] += 0.5
    with pytest.raises(ConfigError, match="spawns"):
        compile_arena(make(tmp_path, spawns=spawns), PHYS)


def test_asymmetric_box_fails(tmp_path):
    walls = {"segments": [], "boxes": [[12.0, 12.0, 4.0, 6.0]]}
    with pytest.raises(ConfigError, match="walls"):
        compile_arena(make(tmp_path, walls=walls), PHYS)


@pytest.mark.parametrize("name", ["ctf_open", "ctf_pillars"])
def test_real_maps(name):
    compile_arena(CONFIGS / "arenas" / f"{name}.yaml", PHYS)


def test_stack_two_maps():
    arenas = [compile_arena(CONFIGS / "arenas" / f"{n}.yaml", PHYS)
              for n in ("ctf_open", "ctf_pillars")]
    s = stack_arenas(arenas)
    assert s.segments.shape == (2, PHYS.max_segments, 4)