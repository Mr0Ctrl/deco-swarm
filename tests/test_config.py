import dataclasses
import shutil
from pathlib import Path

import pytest
import yaml

from deco_swarm.config import ConfigError, load_all, require

REAL = Path(__file__).resolve().parents[1] / "configs"

@pytest.fixture
def cfg_dir(tmp_path):
    for f in REAL.glob("*.yaml"):
        shutil.copy(f, tmp_path)
    return tmp_path

def edit(cfg_dir, name, fn):
    p = cfg_dir / f"{name}.yaml"
    data = yaml.safe_load(p.read_text())
    fn(data)
    p.write_text(yaml.safe_dump(data))


def test_real_configs_load():
    cfg = load_all(REAL)
    assert cfg.agent.body.radius_m == 0.5
    assert cfg.environment.episode.score_limit is None


def test_frozen():
    cfg = load_all(REAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent.body.radius_m = 1.0


def test_hashable():
    hash(load_all(REAL))


def test_unknown_key(cfg_dir):
    edit(cfg_dir, "agent", lambda d: d["body"].update(max_sped_mps=1.0))
    with pytest.raises(ConfigError, match=r"agent\.body.*max_sped_mps"):
        load_all(cfg_dir)


def test_missing_key(cfg_dir):
    edit(cfg_dir, "agent", lambda d: d["body"].pop("radius_m"))
    with pytest.raises(ConfigError, match=r"agent\.body.*radius_m"):
        load_all(cfg_dir)


def test_null_required(cfg_dir):
    edit(cfg_dir, "agent", lambda d: d["tagger"].update(range_m=None))
    with pytest.raises(ConfigError, match=r"agent\.tagger\.range_m: null"):
        load_all(cfg_dir)


def test_null_optional(cfg_dir):
    assert load_all(cfg_dir).environment.episode.score_limit is None


def test_string_number(cfg_dir):
    edit(cfg_dir, "agent", lambda d: d["body"].update(radius_m="0.5"))
    with pytest.raises(ConfigError, match="expected float"):
        load_all(cfg_dir)


def test_int_to_float(cfg_dir):
    edit(cfg_dir, "agent", lambda d: d["body"].update(radius_m=1))
    r = load_all(cfg_dir).agent.body.radius_m
    assert r == 1.0 and type(r) is float


def test_require():
    with pytest.raises(ConfigError):
        require(None, "x")
    assert require(3.0, "x") == 3.0