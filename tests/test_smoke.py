import jax

def test_jax_gpu():
    assert jax.default_backend() == "gpu"

def test_deco_swarm():
    import deco_swarm
    assert isinstance(deco_swarm.__version__, str)

def test_jax_devices():
    assert len(jax.devices()) >= 1