# Faz 1: Minimal CTF Simülasyonu + Viewer (Ayrıntılı Plan)

> Üst plan: [`roadmap_tr.md`](roadmap_tr.md) → Faz 1. Arayüz sözleşmesi: [`../SIM_CONTRACT.md`](../SIM_CONTRACT.md).
> **Hedef:** Elle oynanabilen, GPU'da binlerce kopya halinde paralel koşan bir 2D CTF. Lazer yok, Faz 3'te gelecek.
> **Süre:** ≈2–3 hafta. Alt aşamalar sırayla yapılacak, her biri ayrı bir commit.
> **Çalışma şekli:** Her alt aşamanın bir **çıktısı**, **testleri** ve **bitti kriteri** var.

---

## Bu fazın ana kuralları

1. **Sim içinde Python `if`, `for` ve dinamik shape yok.** Bunların yerine `jnp.where`, maske, lookup tablosu ve `lax.scan` kullanılacak (0.3).
2. **Her fonksiyon tek bir arena için yazılır.** Batch'leme dışarıdan `vmap` ile yapılır.
3. **Birimler metre, saniye ve derece.** `+y` yukarı, açılar `+x` ekseninden saat yönünün tersine ölçülür. Piksel sadece viewer'da kullanılır.
4. **Sayısal değerler config'den gelir.** Kodun içine sabit yazılmaz. Tek istisna sayısal epsilon'lar (`1e-6`).
5. **Bir fizik tick'inin sırası:** `kinematics` → `collision` → `rules`. Bir karar adımı `action_repeat = 4` tick'ten oluşur.

---

## Genel bakış

| # | Alt aşama | Dosya | Çıktı | Tahmini süre |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Kinematik | `sim/kinematics.py` | aksiyon → hareket ve dönüş | 0.5 gün |
| 1.2 | Çarpışma | `sim/collision.py` | duvarda kayma, ajan-ajan itme | 1–2 gün |
| 1.3 | Raycast | `sim/raycast.py` | ışın başına mesafe, sınıf, taşıyıcı biti | 2 gün |
| 1.4 | Oyun kuralları | `sim/state.py`, `sim/rules.py` | `EnvState`, bayrak alma, capture, yerdeki bayrak | 2 gün |
| 1.5 | Env | `envs/ctf_env.py` | `reset`/`step`, gözlem (916), ödül, auto-reset | 2 gün |
| 1.6 | Viewer | `rendering/viewer.py` | pygame, klavyeyle oynama, ışın gösterimi | 1–2 gün |
| 1.7 | SPS + determinizm | `tests/`, `scratch/` | 4096 env'de SPS, aynı seed aynı sonuç | 0.5 gün |

---

## 1.1 Kinematik

**Amaç:** `[move, strafe, turn, fire]` aksiyonunu bir tick'lik (1/60 s) hareket ve dönüşe çevirmek. Çarpışma bu adımda yok.

**Yapılacaklar**
- `decode_actions(actions)`: aksiyonları lookup tablosuyla `(fwd, left, turn, fire)` değerlerine çevirir.
- `integrate(pos, heading_deg, actions, alive, body, dt)`:
  - önce dönüş uygulanır, açı `[0, 360)` aralığında tutulur
  - sonra hareket uygulanır
  - çapraz hareket normalize edilir
  - ölü ajanlar yerinde kalır

**Testler (`tests/test_kinematics.py`)**
- [ ] 0°'de ileri gidince `+x` yönünde ilerliyor. 90°'de ileri gidince `+y` yönünde ilerliyor.
- [ ] 0°'de sola yan adım `+y` yönüne gidiyor.
- [ ] Çapraz hareket tek yöndekinden hızlı değil.
- [ ] Sola dönüş saat yönünün tersine ve 360°'de başa sarıyor.
- [ ] Ölü ajan hareket etmiyor.
- [ ] `jit(vmap(...))` ile `(B, N, ...)` shape'inde çalışıyor.

**Bitti kriteri:** Testler geçiyor.

---

## 1.2 Çarpışma

**Amaç:** Ajanlar duvarın içine giremesin, duvara çarpınca duvar boyunca kaysın, birbirlerinin içine geçmesinler.

**Neden önemli:** Kayma olmazsa ajanlar duvara yapışır ve öğrenme zorlaşır. Bu fonksiyon her tick'te bütün ajanlar × bütün segmentler için çalışacak, bu yüzden SPS'in büyük kısmı burada harcanıyor.

**Yöntem**
1. **Daire–segment:** Ajan merkezine segment üzerindeki en yakın nokta bulunur (projeksiyon + `clip(t, 0, 1)`). Mesafe `d < r` ise ajan normal yönünde `r − d` kadar dışarı itilir. Sadece normal yönünde itildiği için teğet bileşen korunur ve **kayma kendiliğinden** ortaya çıkar.
2. **Vektörel hesap:** `(N, S)` mesafe matrisi bir kerede hesaplanır. `segment_mask` dışındaki satırlar yok sayılır (`jnp.where`).
3. **Köşeler:** İki duvar aynı anda itebilir. `collision_iterations = 2` kez tekrarlanır (`lax.fori_loop` ya da küçük bir Python döngüsü, sayı statik olduğu için ikisi de olur).
4. **Ajan–ajan:** `(N, N)` mesafe matrisi kullanılır. Örtüşme `2r − d > 0` ise iki ajan da yarı yarıya ayrılır. Ölü ajanlar ve ajanın kendisi hesaptan maskeyle çıkarılır.
5. **Tünelleme kontrolü:** Bir tick'teki en büyük hareket 9 m/s × 1/60 s = 0.15 m, ajanın yarıçapı ise 0.5 m. Ajan bir tick'te duvarın içinden geçemez. Hız ya da tick değişirse bu oran yeniden kontrol edilmeli, bu yüzden bir test ekleniyor.

**Testler (`tests/test_collision.py`)**
- [ ] Duvara dik giden ajan duvardan tam `r` uzakta duruyor.
- [ ] Duvara 45° ile giden ajan duvar boyunca kayıyor (teğet hız sıfır değil).
- [ ] İç köşeye giren ajan iki duvarın da içine girmiyor.
- [ ] Maskelenmiş (dolgu) segment ajanı itmiyor.
- [ ] Üst üste başlayan iki ajan simetrik biçimde ayrılıyor.
- [ ] Ölü ajan itilmiyor ve kimseyi itmiyor.
- [ ] `ctf_pillars`'ta 1000 tick rastgele hareketten sonra hiçbir ajan haritanın dışında değil ve hiçbir duvara `r − 1e-3`'ten fazla yaklaşmamış.
- [ ] `linear_speed_mps / tick_hz < radius_m` (tünelleme koruması).

**Bitti kriteri:** Testler geçiyor.

---

## 1.3 Raycast

**Amaç:** Her ajan için `ray_count = 90` ışın gönderip her ışının ilk çarptığı şeyi bulmak: mesafe, sınıf ve çarpılan şeyin bayrak taşıyıcısı olup olmadığı. Gözlemin 900 boyutu buradan geliyor.

**Yöntem**
- Işın açıları: `heading + linspace(−fov/2, +fov/2, R)`.
- **Işın–segment:** Parametrik kesişim hesaplanır. Işının `t > 0` ve segmentin `0 ≤ u ≤ 1` koşulu sağlanmalı. Maskeli segmentlerin mesafesi `inf` sayılır.
- **Işın–daire:** Ajanlar (`radius_m`), bayraklar (`flags.radius_m`) ve üsler (base yarıçapı) için kullanılır. Sadece **dışarıdan giriş** noktası sayılır. Yani ışın bir dairenin içinden başlıyorsa o daireyi görmez. Bu sayede kendi üssündeki ajan üssü görmez ama üssün içindeki bayrağı görür. Kendi kendini görme de bu kuralla engellenir.
- Her ışın için bütün adaylar arasından en küçük mesafe seçilir (`argmin`). Hiçbir şeye çarpmayan ışının sınıfı `none`, mesafesi `max_range_m` olur.
- **Takıma göre sınıflar:** `ally/enemy`, `own_flag/enemy_flag`, `own_base/enemy_base`. Sınıf, bakan ajanın takımına göre belirlenir.
- Ölü ajanlar ışınlara görünmez.
- Çıktı: `dist (N, R)`, `cls (N, R) int`, `carrier (N, R) bool`.

**Karar (SIM_CONTRACT'a yazılacak):** Üsler ve bayraklar ışını durdurur. Işın "dışarıdan giriş" kuralıyla bunları görür.

**Testler (`tests/test_raycast.py`)**
- [ ] Önünde 10 m uzakta duvar olan ajanın orta ışını yaklaşık 10 m ölçüyor ve sınıfı `wall`.
- [ ] Boş yöne giden ışının mesafesi `max_range_m`, sınıfı `none`.
- [ ] Duvarın önünde bir ajan varsa en yakın olan (ajan) kazanıyor.
- [ ] Kırmızı ajan için `ally` olan, mavi ajan için `enemy` görünüyor.
- [ ] Kendi üssünün içindeki ajan üssü görmüyor ama bayrağı görüyor.
- [ ] Ölü ajan görünmüyor.
- [ ] Taşıyıcıya çarpan ışında `carrier = True`.
- [ ] Shape `(N, R)` ve `vmap` ile çalışıyor.

**Bitti kriteri:** Testler geçiyor.

---

## 1.4 Oyun kuralları

**Amaç:** `EnvState`'i tanımlamak ve bir tick'teki CTF kurallarını FTW'ye göre uygulamak. Lazer ve respawn Faz 3'te gelecek, ama state alanları şimdiden yer alıyor.

**`EnvState` (`sim/state.py`, NamedTuple)**
```
pos            (N, 2)   heading_deg   (N,)
alive          (N,)     respawn_timer (N,)     tag_cooldown (N,)   # Phase 3
flag_pos       (2, 2)   flag_state    (2,)  int  0 = home, 1 = carried, 2 = stray
flag_carrier   (2,)  int (-1 = none)           stray_timer  (2,)
score          (2,)  int                       step         ()   int
```
`arena` (0.6'daki `Arena`) state'in içinde değil, yanında taşınıyor. `vmap`'te `in_axes=None` ile paylaşılıyor.

**Bir tick'teki kurallar (sırayla)**
1. **Taşınan bayrak** taşıyıcının pozisyonunu takip eder.
2. **Alma:** Rakip bayrağı `home` ya da `stray` durumundayken, canlı ve bayrak taşımayan bir ajan `radius_m + flag_radius` mesafesine girerse bayrağı alır. Birden çok aday varsa en yakın olan alır.
3. **Geri getirme:** Kendi bayrağı `stray` durumundayken ona dokunan takım arkadaşı bayrağı anında üsse döndürür.
4. **Otomatik dönüş:** `stray_timer ≥ flag_auto_return_s` olursa bayrak üsse döner.
5. **Capture:** Taşıyıcı kendi üs dairesinin içindeyse ve `requires_own_flag_at_home` → kendi bayrağı `home` durumundaysa takıma `score += 1`, taşınan bayrak üsse döner.
6. **Olaylar:** Her ajan için `picked_up`, `returned` ve `captured` bool olarak kaydedilir. Ödül ve log için kullanılır.

Yerdeki bayrak (`stray`) Faz 1'de oyun içinde oluşmuyor, çünkü taşıyıcıyı vuracak lazer yok. Mantık şimdi yazılıyor ve elle kurulmuş bir state ile test ediliyor. Faz 3'te sadece tetikleyicisi (taşıyıcı vurulunca bayrağın düşmesi) eklenecek.

**Testler (`tests/test_rules.py`)** (her test elle kurulmuş küçük bir state kullanıyor)
- [ ] Rakip bayrağına değen ajan bayrağı alıyor. Kendi bayrağı `home` iken ona değmek bir şey değiştirmiyor.
- [ ] İki aday aynı anda değerse sadece en yakın olan alıyor.
- [ ] Taşınan bayrak taşıyıcıyla birlikte hareket ediyor.
- [ ] Üsse dönen taşıyıcı skor yapıyor ve bayrak rakibin üssüne dönüyor.
- [ ] Kendi bayrağı `home` değilken capture olmuyor.
- [ ] Takım arkadaşı yerdeki kendi bayrağına dokununca bayrak üsse dönüyor. Rakip dokunursa alıyor.
- [ ] Yerdeki bayrak `flag_auto_return_s` sonra üsse dönüyor.
- [ ] Olay bayrakları doğru ajanlarda `True`.

**Bitti kriteri:** Testler geçiyor.

---

## 1.5 Env

**Amaç:** Parçaları JaxMARL benzeri bir arayüzde birleştirmek. Faz 2'deki eğitim kodu sadece bu arayüzü görecek.

**Arayüz (`envs/ctf_env.py`)**
```python
env = CTFEnv(configs, arena)
obs, state = env.reset(key)                      # obs: (N, 916)
obs, state, reward, terminated, truncated, info = env.step(key, state, actions)  # actions: (N, 4)
env.agents          # ["red_0", "red_1", "blue_0", "blue_1"]
env.obs_size        # 916
env.action_dims     # (3, 3, 3, 2)
```
JaxMARL'daki `{"agent_0": ...}` dict biçimi Faz 2'de ince bir wrapper ile eklenecek. Çekirdek fonksiyonlar dizilerle çalışıyor.

**Yapılacaklar**
1. **`reset`:** `spawn_mask` içindeki spawn'lar `key` ile karıştırılır, ilk `team_size` tanesi seçilir. Bayraklar `home`, skor 0, `step` 0 olur.
2. **`step`:** `lax.scan` (ya da `fori_loop`) ile `action_repeat` tick boyunca kinematik → çarpışma → kurallar uygulanır. Olaylar tick'ler boyunca OR'lanır. Sonra `step += 1`.
3. **Gözlem:** `SIM_CONTRACT`'taki düzene göre `rays (900) + ego (6) + self (10) = 916`. Hepsi ajanın kendi bakış açısından, takıma göre.
4. **Ödül:** `environment.rewards` config'inden gelen olay ağırlıkları uygulanır. Maç sonunda kazanan takıma `win_reward`. Kesin ağırlıklar Faz 2'de ayarlanacak, burada sadece mekanizma kuruluyor.
5. **Bitiş:** `terminated = step ≥ max_steps` veya (`score_limit` null değilse) `score ≥ score_limit`. `truncated` her zaman False.
6. **Auto-reset:** Bölüm bittiyse aynı `step` çağrısında yeni bir `reset` yapılır ve yeni gözlem döner (`jnp.where` ile, JaxMARL'daki gibi). Bitmiş bölümün son bilgileri `info`'da kalır.

**Testler (`tests/test_env.py`)**
- [ ] `reset` ve `step` doğru shape'leri döndürüyor (`obs (N, 916)`, `reward (N,)`).
- [ ] Gözlemde mesafe ve bearing değerleri `[-1, 1]` aralığında. NaN yok.
- [ ] **Simetri testi:** Simetrik bir başlangıç state'inde `red_0`'ın gözlemi `blue_0`'ınkiyle aynı. Takıma göre gözlemin doğru kurulduğunu kanıtlıyor.
- [ ] `max_steps` sonunda `terminated = True` ve state otomatik sıfırlanıyor.
- [ ] Elle kurulmuş bir capture adımında takımın ödülü pozitif, rakibinki ödül almıyor.
- [ ] `jit(vmap(env.step))` 64 env'de çalışıyor.

**Bitti kriteri:** Testler geçiyor.

---

## 1.6 Viewer

**Amaç:** Sim'i gözle görmek ve klavyeyle oynamak. Kural hatalarının çoğu testte değil, burada fark edilir.

**Bağımlılık:** `uv add pygame` (sadece viewer için, sim kodu import etmiyor).

**Yapılacaklar (`rendering/viewer.py` + `scripts/play.py`)**
- Dünya → ekran dönüşümü: `px = (x − cam_x) · ppm · zoom + W/2`, `py = H/2 − (y − cam_y) · ppm · zoom`. `y` ekranda ters çevriliyor.
- Duvarlar, üsler, bayraklar (durum rengiyle), ajanlar (takım rengi + yön çizgisi) ve skor/süre yazısı çizilir.
- `R` tuşu seçili ajanın ışınlarını aç/kapat yapar. Işınlar sınıf rengiyle çizilir.
- **İnsan modu:**
  - `W/S`: ileri/geri
  - `A/D`: yan adım
  - `←/→`: dönüş
  - `Tab`: kontrol edilen ajanı değiştir
  - diğer ajanlar duruyor ya da rastgele aksiyon alıyor
- Karar adımı 15 Hz, çizim 60 FPS. `env.step` `jit`'li, tek env.
- Bellek: viewer `XLA_PYTHON_CLIENT_PREALLOCATE=false` ile açılır.

**Elle kontrol listesi**
- [ ] Duvar boyunca kayma doğal görünüyor, köşelerde takılma yok.
- [ ] Bayrak alma, taşıma ve capture çalışıyor, skor artıyor.
- [ ] Işınlar duvarlarda, ajanlarda ve bayraklarda doğru sınıfla duruyor.
- [ ] `+y` yukarı: `W` ile 90°'deki ajan ekranda yukarı gidiyor.

**Bitti kriteri:** İki harita da (`ctf_open`, `ctf_pillars`) viewer'da oynanabiliyor.

---

## 1.7 SPS + determinizm

**Amaç:** Sim'in eğitim için yeterince hızlı ve tekrarlanabilir olduğunu ölçmek.

**Yapılacaklar**
- **SPS (`scratch/sps.py`):** 4096 env için `jit(vmap(step))` ve `lax.scan` ile 128 adım, rastgele aksiyonlarla. `SPS = 4096 · 128 / süre` (ilk derleme hariç). 1024, 4096 ve 8192 env'de ölçülür. Sonuç, 0.3'teki nokta-kütle ölçümüyle birlikte not edilir.
- SPS düşükse ilk şüpheli raycast: `N × R × (S + nesne)` hesap. Gerekirse `ray_count`'u düşürmek ya da ışın–segment testini sadeleştirmek düşünülür.
- **Determinizm testi (`tests/test_determinism.py`):** Aynı `key` ile iki kez 200 adımlık rollout yapılır. İki sonucun bütün state yaprakları birebir eşit olmalı (`conftest.py`'deki XLA flag'i etkin). Farklı key ile farklı sonuç çıkmalı.

**Bitti kriteri:** SPS not edildi. Determinizm testi geçiyor.

---

## Faz 1 kapısı (hepsi işaretlenmeden Faz 2'ye geçme)

- [ ] Kinematik, çarpışma, raycast, kural ve env testleri geçiyor.
- [ ] Viewer'da iki harita da elle oynanabiliyor.
- [ ] 4096 env'de SPS ölçüldü ve not edildi.
- [ ] Aynı seed aynı sonucu veriyor.
- [ ] `SIM_CONTRACT.md` uygulamayla uyumlu. Raycast kararı eklendi, değişiklik varsa sürüm artırıldı.

## Faz 2'ye taşınan notlar

- Erken öğrenme için 4500 adım çok uzun olabilir. Faz 2'de ilk denemeler daha kısa bölümle ya da `score_limit: 1` ile yapılabilir (sadece config değişikliği).
- Ödül ağırlıkları (FTW: galibiyet/beraberlik/mağlubiyet + olaylar) Faz 2'de ayarlanacak.
- JaxMARL tarzı dict arayüzü, GRU hafızasının `done`'da sıfırlanması ve minibatch'leme Faz 2'de eğitim koduyla birlikte gelecek.
