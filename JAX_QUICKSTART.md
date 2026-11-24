# DECO-SWARM — JAX Hızlı Başlangıç

Bu branch (`jax-port`) projenin tamamını JAX'a taşır: ortam, model, PPO eğitimi ve viewer.
Davranış eski PyTorch sürümüyle aynıdır: fizik, 640'lık gözlem, ödüller, müfredat ve model mimarisi değişmedi.
Fark: ortamlar GPU'da paralel çalışır ve eğitim döngüsü derlenir (`jit`).

## Kurulum (Ubuntu + NVIDIA GPU)

```bash
git checkout jax-port
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv yoksa
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "jax[cuda12]" flax optax numpy pyyaml pygame
python -c "import jax; print(jax.devices())"        # CudaDevice görünmeli
```

`nvidia-smi` çalışmıyorsa önce sürücüyü kurun: `sudo ubuntu-drivers install`.
Windows'ta JAX yalnızca CPU ile çalışır: `pip install jax flax optax numpy pyyaml pygame`.

## Hız ölçümü

```bash
python benchmark.py --device gpu --train --num-envs 64,256,1024
```

En hızlı `num-envs` değerini eğitimde kullanın.

## Eğitim

```bash
python train.py --device gpu --num-envs 256 --total-timesteps 20000000 --save-interval 1000000
```

- Her güncelleme `num-envs × rollout-steps` adım kullanır (256 × 256 = 65.536). `--total-timesteps` ve `--save-interval` değerlerini buna göre büyük tutun.
- İlk iterasyon derleme yüzünden ~1 dakika sürebilir.
- Eski PyTorch modelinden devam: `--resume checkpoints/model_step_270000.pt` (PyTorch kurulu olmasına gerek yok).
- Tier sabitleme: `--tier 2 --lock-tier`.
- Checkpoint'ler `checkpoints/*.msgpack` olarak kaydedilir.

## İzleme / Oynama

```bash
python play.py --mode watch --model-path checkpoints/model_best.msgpack --map configs/maps/faz_1v1.3.yaml
python play.py --mode human --map configs/maps/faz_1v0.1.yaml
```

Kontroller: W/S ileri/geri, A/D yana, Q/E dönüş, fare tekerleği zoom, sağ tık sürükle kaydır, C takip, F1 ray'ler, F2 reset.

## Klasörler

| Yol | İçerik |
|---|---|
| `deco/` | JAX kodu: `env.py`, `model.py`, `ppo.py`, `curriculum.py`, `checkpoint.py`, `rendering/` |
| `train.py`, `play.py`, `benchmark.py` | Giriş noktaları |
| `tests/` | Testler; eski sürümle adım adım karşılaştırma dahil |
| `legacy/` | Eski PyTorch sürümü (`python legacy/train.py`) |

## Testler

```bash
pip install pytest
python -m pytest tests -q
```

Eski sürümle karşılaştırma testleri ek olarak `gymnasium pettingzoo torch` ister. Bunlar kurulu değilse o testler atlanır.
