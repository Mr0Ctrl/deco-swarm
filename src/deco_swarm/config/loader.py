import dataclasses
import types
import typing
from pathlib import Path

import yaml
from .schema import Configs

class ConfigError(ValueError):
    pass


def _convert(tp, value, path):
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    
    if isinstance(tp, types.UnionType):
        if value is None and type(None) in typing.get_args(tp):
            return None
        (tp, ) = [a for a in typing.get_args(tp) if a is not type(None)]
    
    if value is None:
        raise ConfigError(f"{path}: null but required")
    
    if tp is float and type(value) is int:
        return float(value)

    if type(value) is not tp:
        raise ConfigError(f"{path}: expected {tp.__name__}, got {type(value).__name__} ({value!r})")
    return value

def _build(cls, data, path):
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    unknown = data.keys() - hints.keys()
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    missing = hints.keys() - data.keys()
    if missing:
        raise ConfigError(f"{path}: missing keys {sorted(missing)}")
    return cls(**{name: _convert(tp, data[name], f"{path}.{name}") for name, tp in hints.items()})

def load_all(config_dir) -> Configs:
    config_dir = Path(config_dir)
    sections = {}
    for name, cls in typing.get_type_hints(Configs).items():
        with open(config_dir / f"{name}.yaml", encoding="utf-8") as f:
            sections[name] = _build(cls, yaml.safe_load(f), name)
    
    return Configs(**sections)

def require(value, name):
    if value is None:
        raise ConfigError(f"{name}: pending decision (null), must be set before use")
    return value