import os

flag = "--xla_gpu_deterministic_ops=true"
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()