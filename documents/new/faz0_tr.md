# Faz 0: Temel ve Öğrenme (Ayrıntılı Plan)

> Üst plan: [`roadmap_tr.md`](roadmap_tr.md) → Faz 0.
> **Hedef:** Araçları tanımak, repo iskeletini kurmak, config'leri ve arena derleyiciyi yazmak, JaxMARL'ın MAPPO kodunu anlayıp bizim GPU'da çalıştırmak.
> **Süre:** ≈1–2 hafta. Alt aşamalar sırayla gidiyor. 0.3 (JAX alıştırmaları) ile 0.7 (JaxMARL) diğerleriyle paralel yürütülebilir.
> **Çalışma şekli:** Her alt aşamanın bir **çıktısı** ve bir **bitti kriteri** var. Kriter sağlanınca o aşamayı ayrı bir commit olarak at.

---

## Genel bakış

| # | Alt aşama | Çıktı | Tahmini süre |
| :--- | :--- | :--- | :--- |
| 0.1 | Python + JAX ortamı | `.venv`, GPU'yu gören JAX | 0.5 gün |
| 0.2 | Repo iskeleti | `pyproject.toml`, paket düzeni, boş testler geçiyor | 0.5 gün |
| 0.3 | JAX temelleri (alıştırma) | `scratch/` altında küçük betikler | 2–3 gün |
| 0.4 | Config dosyaları (JSON) | `configs/*.json`, `configs/arenas/*.json` | 1 gün |
| 0.5 | Config yükleyici | `config/` modülü + testleri | 1–2 gün |
| 0.6 | Arena derleyici | Segment dizisi + maske + simetri kontrolü + testleri | 1–2 gün |
| 0.7 | JaxMARL `mappo_rnn` incelemesi | GPU'da eğitim + okuma notları | 2–3 gün |
| 0.8 | PPO/GAE ve ilk eğitim parçası | `training/gae.py` + NumPy referansına karşı test | 1 gün |
| 0.9 | `SIM_CONTRACT.md` v0 taslağı | Birim, aksiyon ve gözlem sözleşmesi | 0.5 gün |

---

## 0.1 — Python + JAX ortamı

**Mevcut durum:** RTX 4060 Laptop (8 GB), sürücü 535 (CUDA 12.2), sistem Python 3.10, `uv` kurulu.

**Yapılacaklar**
1. Python 3.12 kullan: `uv python install 3.12`. Güncel JAX sürümleri Python 3.10 desteğini bıraktı, ayrıca sistem Python'una dokunmamış oluruz.
2. Proje kökünde sanal ortam aç (`uv venv --python 3.12`).
3. Temel paketler (öneri, eklemeden önce birlikte karar verelim):
   - `jax[cuda12]`: simülasyon ve eğitim
   - `flax`: ağlar (JaxMARL de bunu kullanıyor)
   - `optax`: optimizer
   - `numpy`, `pytest`
   - Sonraya bırakılanlar: `pygame` (Faz 1 viewer), `orbax-checkpoint` ve `tensorboardX` (Faz 2)
4. GPU kontrolü:
   ```bash
   python -c "import jax; print(jax.devices()); print(jax.numpy.ones(3).device)"
   ```
   Çıktıda `CudaDevice` görmelisin. `CpuDevice` görüyorsan veya `ptxas`/sürücü uyarısı çıkıyorsa NVIDIA sürücüsünü 550+ sürümüne güncelle.
5. Bellek ayarını öğren: JAX varsayılan olarak GPU belleğinin %75'ini baştan ayırır. Viewer ile eğitim aynı anda açıksa sorun olur. Gerekirse `XLA_PYTHON_CLIENT_PREALLOCATE=false` veya `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` kullan.

**Bitti kriteri:** `jax.devices()` GPU'yu gösteriyor. 4096×4096 matris çarpımı CPU'dakinden belirgin şekilde hızlı (alıştırma olarak ölç, `block_until_ready()` kullanmayı unutma).

> Not: İkinci laptopta (RTX 3070) da aynı kurulum yapılmalı. Paket sürümleri `uv.lock` ile sabitlenince iki makine aynı ortamı kullanır.

---

## 0.2 — Repo iskeleti

**Önerilen düzen** (varsayılan olarak bunu seçtim: `src/sim` yerine `src/deco_swarm/sim`, çünkü `sim` ve `envs` gibi genel isimler başka paketlerle çakışabilir):

```
deco-swarm/
├── pyproject.toml
├── uv.lock
├── .gitignore            # .venv/, __pycache__/, runs/, checkpoints/, scratch/
├── configs/
│   ├── agent.json
│   ├── physics.json
│   ├── observation.json
│   ├── game_rules.json
│   ├── rewards.json
│   ├── render.json
│   └── arenas/
│       └── ctf_open_v0.json
├── src/deco_swarm/
│   ├── __init__.py
│   ├── config/           # yükleyici + dataclass'lar (0.5) + arena derleyici (0.6)
│   ├── sim/              # Faz 1
│   ├── envs/             # Faz 1
│   ├── models/           # Faz 2
│   ├── training/         # 0.8'de gae.py ile başlıyor
│   ├── rendering/        # Faz 1
│   └── eval/             # Faz 3+
├── tests/
│   └── test_smoke.py
├── scratch/              # alıştırmalar, git'e girmez
└── documents/
```

**Yapılacaklar**
1. `pyproject.toml`: proje adı, Python sürümü (`>=3.12`), bağımlılıklar, `[tool.pytest.ini_options]` içinde `testpaths = ["tests"]`.
2. Paketi düzenlenebilir modda kur (`uv pip install -e .` veya `uv sync`). Böylece testlerde `from deco_swarm.config import ...` çalışır.
3. `tests/test_smoke.py`: paketin import edildiğini ve `jax.devices()` çağrısının hata vermediğini kontrol eden tek bir test.
4. Determinizm notu: Faz 1'de "aynı seed aynı sonuç" testi gelecek. GPU'da bazı işlemler deterministik değil. Bunun için `XLA_FLAGS=--xla_gpu_deterministic_ops=true` ayarını şimdiden `tests/conftest.py` içinde (import'tan önce `os.environ` ile) ayarlamak iyi bir alışkanlık.

**Bitti kriteri:** `pytest` çalışıyor ve smoke testi geçiyor.

---

## 0.3 — JAX temelleri (alıştırma)

Bu aşamanın amacı kod üretmek değil, **sezgi kazanmak**. Her alıştırmayı `scratch/` altında küçük bir betik olarak yaz. Sonucu tahmin et, çalıştır, tahminle karşılaştır.

### A. `jit` ve tracing
- `jit`'li bir fonksiyonun içine `print(x)` koy ve iki kez çağır. Kaç kez yazdırıyor, ne yazdırıyor? (Cevap: sadece ilk çağrıda, tracer nesnesi yazdırır.) `jax.debug.print` ile farkını gör.
- Fonksiyon içinde `if x > 0:` yaz ve `jit` ile çalıştır. Hatayı oku. Sonra aynı mantığı `jnp.where` ve `jax.lax.cond` ile yaz.
- Aynı fonksiyonu farklı shape'lerle çağır ve yeniden derlemeyi gözlemle. **Ders:** Sabit shape, Faz 1'deki "padding + maske" kararının sebebi.
- `static_argnums` kullan: bir tamsayıyı statik geçince ne değişiyor?

### B. `vmap`
- Tek bir 2D noktayı bir duvar segmentine göre iten fonksiyon yaz: `push(pos, seg) -> pos`. Önce bir noktayla çalıştır, sonra `vmap` ile 4096 noktaya, sonra iç içe `vmap` ile 4096 nokta × 64 segmente uygula.
- `in_axes` ile oyna: segmentler bütün env'lerde ortak, pozisyonlar env başına ayrı olsun (`in_axes=(0, None)`).

### C. `lax.scan`
- Basit bir 1D nokta-kütle ortamı yaz: `state = (pos, vel)`, aksiyon = ivme. `step(state, action) -> (new_state, reward)`.
- 100 adımlık rollout'u önce Python `for` döngüsüyle, sonra `lax.scan` ile yaz. İkisinin `jit` derleme süresini ve çalışma süresini karşılaştır.
- Sonra bunu `vmap` ile 4096 env'e çıkar ve **SPS** (saniyedeki adım) ölç. Faz 1'deki SPS ölçümünün küçük bir provası bu.

### D. PyTree
- `EnvState` benzeri bir yapıyı üç şekilde dene: `NamedTuple`, `flax.struct.dataclass`, düz `dict`. `jax.tree_util.tree_map(lambda x: x.shape, state)` ile shape'leri yazdır.
- `vmap` edilmiş bir state'in yapraklarının başına batch ekseni eklendiğini gör.
- Frozen standart `dataclass` bir PyTree **değildir**. `jit`'e argüman olarak verince ne oluyor? (0.5'teki config tasarımı için önemli.)

### E. PRNG anahtarları
- `key = jax.random.key(0)`; `jax.random.split` ile anahtar böl. Aynı anahtarı iki kez kullanınca aynı sayıyı aldığını gör. **Kural:** Bir anahtar bir kez kullanılır.
- 4096 env için anahtarları `split(key, 4096)` ile üret ve `vmap`'li bir reset'e ver.
- `lax.scan` içinde anahtarı carry'de taşı ve her adımda böl.

**Bitti kriteri:** Şu sorulara kodsuz cevap verebiliyorsun:
1. Neden `jit` içinde Python `if` kullanamıyoruz?
2. `vmap` ve `lax.scan` arasındaki fark ne, hangisi paralel, hangisi sıralı?
3. Bir config nesnesi `jit`'e statik mi geçmeli, PyTree olarak mı? Neden?
4. Rollout içinde PRNG anahtarı nasıl taşınır?

---

## 0.4 — Config dosyaları (JSON)

**İlkeler**
- Birimler metre / saniye / derece. Anahtar adında birimi yaz: `radius_m`, `max_speed_mps`, `turn_rate_dps`, `fov_deg`, `tick_hz`. Bu, birim hatalarını okuma sırasında yakalatır.
- Henüz karar verilmemiş alanlar `null`. Bu alanlar şemada da "bekleyen karar" olarak işaretlenecek (bkz. 0.5).
- Başlangıç değerleri: v1 dokümanındaki piksel değerleri 1000 px → 50 m (×0.05) oranıyla çevrildi. Bunlar Faz 1'de viewer'da oynayarak ayarlanacak.

**Önerilen içerik (taslak)**

`agent.json`
```json
{
  "body": { "radius_m": 0.5, "max_speed_mps": 9.0, "turn_rate_dps": 200.0 },
  "perception": { "ray_count": 90, "fov_deg": 180.0, "max_range_m": 25.0 }
}
```

`physics.json`
```json
{ "tick_hz": 60.0, "action_repeat": 6, "collision_iterations": 2, "max_segments": 64 }
```

`observation.json`
```json
{
  "ray_classes": ["none", "wall", "ally", "enemy", "own_flag", "enemy_flag", "own_base", "enemy_base"],
  "ray_hit_carrier_bit": true,
  "ego_bearings": ["own_base", "enemy_flag_stand"],
  "self_state": ["carrying_flag", "own_flag_stolen", "in_own_half", "respawn_timer"]
}
```

`game_rules.json` (açık noktalar burada `null`)
```json
{
  "episode": { "max_decision_steps": 300 },
  "flag": { "pickup_radius_m": 1.0, "dropped_flag_returns_to_stand": true },
  "capture": { "requires_own_flag_at_home": null },
  "laser": {
    "enabled": false,
    "range_m": null,
    "cooldown_s": null,
    "only_in_own_half": null,
    "carrier_can_fire": null
  },
  "respawn": { "delay_s": null, "protection_s": null }
}
```

`rewards.json`, `render.json`: aynı mantıkla. Shaping ağırlıkları şimdilik `null` (Faz 2'de doldurulacak). Render'da pencere boyutu ve `pixels_per_meter` olsun. Piksel dönüşümü **sadece** viewer'da yapılacak.

`arenas/ctf_open_v0.json`
```json
{
  "name": "ctf_open_v0",
  "size_m": [50.0, 30.0],
  "symmetry": "mirror_x",
  "walls": {
    "segments": [ [[25.0, 5.0], [25.0, 10.0]] ],
    "boxes":    [ [12.0, 12.0, 4.0, 6.0] ]
  },
  "teams": {
    "red":  { "base": { "center": [4.0, 15.0], "radius_m": 3.0 },
              "flag_stand": [4.0, 15.0],
              "spawns": [[6.0, 12.0, 0.0], [6.0, 18.0, 0.0]] },
    "blue": { "base": { "center": [46.0, 15.0], "radius_m": 3.0 },
              "flag_stand": [46.0, 15.0],
              "spawns": [[44.0, 12.0, 180.0], [44.0, 18.0, 180.0]] }
  }
}
```
Not: Bu örnekte `boxes` simetrik değil. Simetri kontrolünü test etmek için bilerek böyle bıraktım. Gerçek haritada mavi taraf için aynalanmış kutuyu da ekle.

**Bitti kriteri:** Tüm dosyalar geçerli JSON (`python -m json.tool dosya.json`). Her `null` alan [`roadmap_tr.md`](roadmap_tr.md) → "Açık noktalar" listesindeki bir maddeye karşılık geliyor.

---

## 0.5 — Config yükleyici

**Tasarım kararları**
1. **Frozen dataclass**, iç içe yapı: `AgentConfig(body=BodyConfig(...), perception=PerceptionConfig(...))`. Listeler yerine `tuple` kullan, aksi halde nesne hash'lenemez.
2. **Config'ler `jit`'e statik geçer.** Bunlar shape'leri ve sabitleri belirliyor (`ray_count`, `max_segments`...). Hash'lenebilir frozen dataclass bunun için uygun. Dizi verisi (arena segmentleri) config değildir: 0.6'da ayrı bir PyTree olarak üretilecek.
3. **Katı doğrulama:**
   - JSON'da olup dataclass'ta olmayan anahtar → hata (yazım hatası koruması: `max_sped_mps`).
   - Dataclass'ta olup JSON'da olmayan anahtar → hata.
   - Tip uyuşmazlığı → hata. `int` → `float` dönüşümüne izin ver, `"9.0"` (string) → hata.
4. **`null` politikası:**
   - Tipi `float` olan alan `null` gelirse → hata: `"game_rules.laser.range_m: null ama zorunlu"` gibi tam yol içeren mesaj.
   - Tipi `float | None` olan alan "bekleyen karar" demek, `null` kabul edilir.
   - Kullanım yerinde küçük bir `require(value, name)` yardımcı fonksiyonu: değer `None` ise anlaşılır bir hata verir. Böylece Faz 1, Faz 3 kararlarını beklemeden çalışır. Faz 3'te karar verilince tip `float`'a çevrilir ve yükleyici artık `null`'a izin vermez.
5. **Tek giriş noktası:** `load_all(config_dir) -> Configs` hepsini bir kerede yükler. Simülasyon kodu dosya okumaz, sadece bu nesneyi alır.
6. (Opsiyonel, sonraya bırakılabilir) CLI override: `--set physics.tick_hz=30`. Öncelik: CLI > JSON. Kod içinde varsayılan değer tutmuyoruz, çünkü kaynak tek olsun.

**Yazılacak testler (`tests/test_config.py`)**
- [ ] Repodaki gerçek config'ler hatasız yükleniyor.
- [ ] Nesneye atama `FrozenInstanceError` veriyor.
- [ ] `hash(configs)` çalışıyor.
- [ ] Bilinmeyen anahtar → hata, mesajda anahtarın tam yolu var.
- [ ] Eksik anahtar → hata.
- [ ] Zorunlu alanda `null` → hata. Opsiyonel alanda `null` → `None`.
- [ ] String verilen sayısal alan → hata. `int` verilen `float` alanı → `float`.
- [ ] `require(None, "x")` hata veriyor, `require(3.0, "x")` değeri döndürüyor.

Testler için geçici JSON dosyalarını `tmp_path` fixture'ı ile üret. Gerçek config'leri bozarak test yazma.

**Bitti kriteri:** Tüm testler geçiyor.

---

## 0.6 — Arena derleyici

**Girdi:** `ArenaConfig` (0.5'teki gibi JSON'dan yüklenmiş). **Çıktı:** sabit shape'li bir PyTree:

```
Arena
├── segments       float32 (MAX_SEGMENTS, 4)   # [x1, y1, x2, y2], dolgu satırları 0
├── segment_mask   bool    (MAX_SEGMENTS,)
├── base_center    float32 (2, 2)              # [takım, xy]  0 = red, 1 = blue
├── base_radius    float32 (2,)
├── flag_stand     float32 (2, 2)
├── spawns         float32 (2, MAX_PER_TEAM, 3) # [x, y, heading_deg]
└── size           float32 (2,)
```

**Adımlar**
1. Dış sınır: `[0, W] × [0, H]` için 4 segment otomatik eklenir.
2. Her kutu `[x, y, w, h]` → 4 segment.
3. Serbest segmentler olduğu gibi eklenir.
4. Toplam `MAX_SEGMENTS`'i aşarsa → hata (sessizce kesme).
5. Kalan satırlar sıfırla doldurulur, maske `False`.
6. Simetri kontrolü (`symmetry: "mirror_x"`): her kırmızı öğenin `(x, y, h) → (W − x, y, 180 − h)` aynası mavi tarafta bulunmalı (spawn, üs, kaide). Duvar kümesi de kendi aynasına eşit olmalı (segment yönü önemsiz, `[a, b]` ile `[b, a]` aynı). Küçük tolerans (`1e-5`) kullan.
7. Basit geçerlilik kontrolleri: spawn ve kaideler harita içinde, iki takımın spawn sayısı eşit.

Bu fonksiyon NumPy ile yazılabilir. Yükleme sırasında bir kez çalışıyor, `jit` gerekmez. Sonunda `jnp.asarray` ile diziye çevir.

**Yazılacak testler (`tests/test_arena.py`)**
- [ ] Duvarsız harita → maskede tam 4 `True`.
- [ ] Bir kutu → 4 + 4 = 8 segment, kutunun köşeleri doğru.
- [ ] Shape'ler her zaman `(MAX_SEGMENTS, 4)` ve `(MAX_SEGMENTS,)`, segment sayısından bağımsız.
- [ ] `MAX_SEGMENTS` aşılınca hata.
- [ ] Simetrik arena geçiyor. Bir spawn'ı 0.5 m kaydırınca hata veriyor.
- [ ] (Bilerek asimetrik bıraktığın) `ctf_open_v0` örneği simetri hatası veriyor, düzeltince geçiyor.
- [ ] İki farklı arena derlenip `jnp.stack` ile birleştirilebiliyor (shape'ler aynı). Bu, ileride `vmap` ile farklı haritalarda eğitimin ön koşulu.

**Bitti kriteri:** Testler geçiyor. Ek olarak, matplotlib ile segmentleri, üsleri ve spawn'ları çizen küçük bir `scratch/` betiğiyle haritayı gözle kontrol et (matplotlib bağımlılığı sadece scratch için, istersen).

---

## 0.7 — JaxMARL `mappo_rnn` incelemesi

**Kurulum (bizim ortamı kirletmeden)**
- JaxMARL kendi JAX sürümünü sabitleyebiliyor. Bu yüzden repoyu **proje dışına** klonla (ör. `~/refs/JaxMARL`) ve ona ayrı bir sanal ortam aç.
- `baselines/MAPPO/` altındaki RNN'li MAPPO betiğini bul (dosya adları sürüme göre değişebilir, ör. `mappo_rnn_mpe.py`). Ayrıca karşılaştırma için `baselines/IPPO/` altındaki RNN'li IPPO betiğine bak.
- Önce küçük bir MPE ortamında (ör. `simple_spread`) çalıştır. Gerekirse env sayısını ve toplam adımı düşür. WandB'yi kapat (`WANDB_MODE=disabled` veya config'deki ilgili alan).

**Okurken cevaplanacak sorular** (cevapları `documents/notes/jaxmarl_notlari.md` dosyasına yaz, Faz 2'de buna dönülecek)
1. Ortam nasıl `vmap`'leniyor? Auto-reset nerede yapılıyor?
2. Rollout `lax.scan`'ının carry'sinde neler var?
3. GRU hidden state'i `done` olduğunda nerede sıfırlanıyor? Rollout'ta mı, eğitimde mi, ikisinde de mi?
4. Eğitim sırasında hidden state nereden başlatılıyor (chunk başındaki saklı state)?
5. GAE nerede, `lax.scan` ile nasıl ve hangi yönde hesaplanıyor?
6. Minibatch'ler hangi eksende bölünüyor? Neden zaman ekseni bölünmüyor?
7. MAPPO'da critic'e giden global state nasıl kuruluyor, IPPO'dan farkı ne?
8. Advantage normalizasyonu, gradyan kırpma, entropi katsayısı nerede?
9. `terminated` ile `truncated` ayrımı yapılıyor mu? (Yapılmıyorsa bunu not et, Faz 2'de bizim yapmamız gerekecek.)
10. Bizim CTF ortamımızın `MultiAgentEnv` arayüzüne uyması için hangi metotlar ve özellikler gerekli? (`reset`, `step`, `observation_space`, `action_space`, `agents`...)

**Ölçüm**
- Eğitimin toplam süresi ve SPS'i not et. `nvidia-smi` ile GPU kullanımını ve bellek tüketimini izle.
- Env sayısını 2 katına çıkar ve SPS'in nasıl değiştiğine bak.

**Bitti kriteri:** Örnek bizim GPU'da sonuna kadar eğitiyor ve ödül eğrisi artıyor. 10 sorunun cevabı not dosyasında.

---

## 0.8 — PPO/GAE ve ilk eğitim parçası

**Önce sezgi:** PPO, "politikayı iyi sonuç veren aksiyonlara doğru it, ama bir güncellemede çok uzaklaşma" demek. GAE ise "bu aksiyon beklenenden ne kadar iyiydi?" sorusunun gürültüsü az bir tahmini. 1 adımlık TD hatası (düşük varyans, yüksek bias) ile tam Monte Carlo getirisi (yüksek varyans, düşük bias) arasında `λ` ile ayar yapar.

**Matematik (kısaca)**
- δₜ = rₜ + γ·V(sₜ₊₁)·(1 − doneₜ) − V(sₜ)
- Âₜ = δₜ + γλ·(1 − doneₜ)·Âₜ₊₁ (sondan başa doğru hesaplanır)
- Hedef değer: Rₜ = Âₜ + V(sₜ)

**Yapılacaklar**
1. `src/deco_swarm/training/gae.py`: `compute_gae(rewards, values, dones, last_value, gamma, lam)` fonksiyonunu `lax.scan(..., reverse=True)` ile yaz. Girdiler `(T, B)` shape'inde.
2. Testte aynı hesabı düz NumPy `for` döngüsüyle yap ve iki sonucu `np.testing.assert_allclose` ile karşılaştır.
3. Ek test vakaları:
   - [ ] `λ = 0` → Â = δ (saf TD).
   - [ ] `λ = 1`, `done` yok → Â + V = indirgenmiş getiri.
   - [ ] Ortada bir `done` varsa, sonraki bölümün ödülü öncekine sızmıyor.
4. Düşünme sorusu: `truncated` (süre doldu) durumunda `(1 − done)` ile değeri sıfırlamak neden yanlış? Doğru davranış nedir? Cevabı not al. Faz 2'nin konusu, ama formülün nerede değişeceğini şimdiden bil.

**Bitti kriteri:** GAE testleri geçiyor ve JaxMARL'daki GAE ile kendi yazdığının aynı işi yaptığını açıklayabiliyorsun.

---

## 0.9 — `SIM_CONTRACT.md` v0 taslağı

Faz 1'e başlamadan önce simülasyonun "dış yüzünü" yaz. Kod yok, sadece sözleşme. Viewer, eğitim ve ileride görsel motora aktarım bu dokümana bakacak.

**İçerik**
- **Sürüm:** `v0`. Gözlem veya aksiyon şeması değişirse sürüm artar. Checkpoint'lere bu sürüm yazılacak.
- **Birimler ve koordinat sistemi:** metre, saniye, derece. Orijin sol alt köşe, `+x` sağ, `+y` yukarı, açı `+x`'ten saat yönünün tersine.
- **Zamanlama:** 60 Hz fizik, 10 Hz karar (`action_repeat = 6`).
- **Aksiyon uzayı:** hareket (5: dur/ileri/geri/sol/sağ) × dönüş (3: sol/yok/sağ), multi-discrete. Lazer `fire` aksiyonu Faz 3'te eklenecek (sürüm artacak).
- **Gözlem şeması:** ışın başına `[mesafe/menzil, sınıf one-hot (8), taşıyıcı biti]`, ego yönleri (kendi üs, rakip kaide: her biri `cos, sin, mesafe`), öz-durum alanları. Toplam boyut formülünü yaz.
- **Ajan sıralaması:** `agents = ["red_0", "red_1", ..., "blue_0", ...]`. Dizilerde takım ekseni ve ajan indeksi bu sıraya göre.
- **Bölüm sonu:** `terminated` (capture) ve `truncated` (süre doldu) ayrı döner.

**Bitti kriteri:** Doküman `documents/SIM_CONTRACT.md` olarak repoda ve 0.4'teki `observation.json` ile çelişmiyor.

---

## Faz 0 kapısı (hepsi işaretlenmeden Faz 1'e geçme)

- [ ] `jax.devices()` iki laptopta da GPU gösteriyor ve ortam `uv.lock` ile sabit.
- [ ] `pytest` geçiyor: smoke, config, arena ve GAE testleri.
- [ ] `configs/` altındaki tüm `null`'lar "Açık noktalar" listesiyle eşleşiyor.
- [ ] JaxMARL MAPPO-RNN örneği bizim GPU'da eğitiyor. SPS ve bellek notu alındı.
- [ ] `documents/notes/jaxmarl_notlari.md` 10 soruyu cevaplıyor.
- [ ] `documents/SIM_CONTRACT.md` v0 yazıldı.
- [ ] 0.3'teki dört soruya kodsuz cevap verilebiliyor.

## Faz 1'e taşınan notlar

- 0.3-C'deki nokta-kütle SPS ölçümü, Faz 1'deki SPS hedefi için bir alt sınır fikri verir.
- 0.6'daki `Arena` PyTree'si doğrudan `EnvState`'in statik kısmı olacak.
- 0.7'deki 10. soru, Faz 1'deki `MultiAgentEnv` arayüzünün kontrol listesi.
