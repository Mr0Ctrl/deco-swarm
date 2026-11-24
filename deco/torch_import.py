# deco/torch_import.py
"""
Loads checkpoints written by the legacy PyTorch code WITHOUT needing PyTorch installed,
and converts the RecurrentActorCritic state_dict into Flax parameters.

A torch .pt file is a zip archive holding a pickle (data.pkl) whose tensors point at raw storage
files (data/<key>). Only the handful of types a state_dict needs are allowed when unpickling,
so arbitrary code in a malicious file cannot run (same idea as torch.load(weights_only=True)).
The file may also be given as an already-extracted directory.
"""
import os
import pickle
import zipfile
from collections import OrderedDict
from typing import Any, Callable, Dict

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        if name not in _STORAGE_DTYPES:
            raise pickle.UnpicklingError(f"Unsupported torch storage type: {name}")
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    if len(size) == 0:
        return storage[storage_offset].copy()
    itemsize = storage.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[storage_offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    ).copy()


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
    return data


class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, file, read_record: Callable[[str], bytes]):
        super().__init__(file)
        self._read_record = read_record

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        raise pickle.UnpicklingError(f"Refusing to load '{module}.{name}' from a checkpoint")

    def persistent_load(self, pid):
        typename, storage_type, key, _location, _numel = pid
        if typename != "storage":
            raise pickle.UnpicklingError(f"Unknown persistent id: {typename}")
        return np.frombuffer(self._read_record(f"data/{key}"), dtype=storage_type.dtype)


def _find_extracted_root(path: str) -> str:
    """Accepts the directory that holds data.pkl, or a parent containing exactly one such directory."""
    if os.path.exists(os.path.join(path, "data.pkl")):
        return path
    subdirs = [d for d in os.listdir(path) if os.path.exists(os.path.join(path, d, "data.pkl"))]
    if len(subdirs) != 1:
        raise FileNotFoundError(f"No torch checkpoint (data.pkl) found under {path}")
    return os.path.join(path, subdirs[0])


def load_torch_checkpoint(path: str) -> Any:
    """Reads a torch.save() file (zip) or its extracted directory into numpy arrays / python containers."""
    if os.path.isdir(path):
        root = _find_extracted_root(path)

        def read_record(name: str) -> bytes:
            with open(os.path.join(root, *name.split("/")), "rb") as f:
                return f.read()

        with open(os.path.join(root, "data.pkl"), "rb") as f:
            return _TorchUnpickler(f, read_record).load()

    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
        prefix = pkl_name[: -len("data.pkl")]

        def read_record(name: str) -> bytes:
            return zf.read(prefix + name)

        with zf.open(pkl_name) as f:
            return _TorchUnpickler(f, read_record).load()


def torch_state_dict_to_flax(sd: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Maps legacy RecurrentActorCritic parameter names/layouts onto deco.model.RecurrentActorCritic."""

    def linear(prefix):
        return {"kernel": np.asarray(sd[f"{prefix}.weight"]).T, "bias": np.asarray(sd[f"{prefix}.bias"])}

    def conv(prefix):
        # torch Conv1d weight (out, in, k) -> flax Conv kernel (k, in, out)
        return {"kernel": np.transpose(sd[f"{prefix}.weight"], (2, 1, 0)), "bias": np.asarray(sd[f"{prefix}.bias"])}

    params = {
        "spatial_encoder": {
            "conv1": conv("spatial_encoder.conv_net.0"),
            "conv2": conv("spatial_encoder.conv_net.2"),
            "fc": linear("spatial_encoder.conv_net.6"),
        },
        "ego_encoder": {
            "fc1": linear("ego_encoder.mlp.0"),
            "fc2": linear("ego_encoder.mlp.2"),
        },
        "gru": {
            "input": {"kernel": np.asarray(sd["gru_cell.weight_ih"]).T, "bias": np.asarray(sd["gru_cell.bias_ih"])},
            "hidden": {"kernel": np.asarray(sd["gru_cell.weight_hh"]).T, "bias": np.asarray(sd["gru_cell.bias_hh"])},
        },
        "actor_fc": linear("actor_backbone.0"),
        "move_head": linear("move_head"),
        "rot_head": linear("rot_head"),
        "critic_fc": linear("critic_head.0"),
        "critic_out": linear("critic_head.2"),
    }
    return {"params": params}
