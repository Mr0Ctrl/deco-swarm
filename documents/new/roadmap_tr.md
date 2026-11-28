# DECO-SWARM v2 — Yol Haritası

> **İlke:** Simülasyonu tamamen bitirip sonra eğitime geçmek yerine, her fazda **çalışan ve eğitilebilen** bir sürüm. Kurallar ve sabitler `configs/*.json` içinde. `null` olan alanlar henüz karar verilmemiş noktalar.
> **Öğrenme:** Her fazın bir "anlamamız gerekenler" listesi var. Kodu birlikte yazarken bu konuları da sindiriyoruz.

## Alınmış kararlar

| Konu | Karar |
| :--- | :--- |
| Dil / altyapı | Saf JAX. PPO/MAPPO için JaxMARL referans alınıp uyarlanacak. |
| Birimler | Metre / saniye / derece |
| İletişim | Türevlenebilir mesajlar (DIAL/CommNet tarzı), 1 adım gecikme, takım içi |
| Tag | Lazerle vurma (ayrı `fire` aksiyonu, cooldown) |
| Düşen bayrak | Taşıyıcı vurulunca bayrak kaideye döner |
| Görsel | Güzel görüntü için kayıt → başka motorda replay. İnsan vs. bot için ayrıca canlı oynanış (Faz 5). |
| Config formatı | JSON (RS IR-DATA-01'de "YAML" yazıyor, güncellenmeli) |

---

## Faz 0 — Temel ve öğrenme (≈1–2 hafta)

**Hedef:** Araçları tanımak, iskeleti kurmak.
- Repo düzeni: `src/{sim,envs,models,training,rendering,eval}`, `tests/`, `configs/`
- JSON yükleyici: frozen dataclass'lara çevirsin, `null` kalan zorunlu alanlarda hata versin.
- JaxMARL'ın `mappo_rnn` kodunu satır satır oku ve küçük bir ortamında çalıştır.

**Kapı:** Config testleri geçiyor. JaxMARL örneği bizim GPU'da eğitiyor.
**Anlamamız gerekenler:** `jit`, `vmap`, `lax.scan`, PyTree, PRNG key yönetimi; PPO ve GAE'nin mantığı.

---

## Faz 1 — Minimal CTF simülasyonu + viewer (≈2–3 hafta)

**Hedef:** Elle oynanabilen, hızlı bir 2D CTF.
- `EnvState`, kinematik, kayan duvar çarpışması, ajan-ajan itme
- Semantik raycast (`observation.json`'daki sınıflar)
- Bayrak alma, üsse getirme, capture → bölüm sonu. **Lazer henüz yok.**
- pygame viewer: insan modu, ışın gösterimi
- JaxMARL `MultiAgentEnv` arayüzü

**Kapı:** Çarpışma/raycast/kural unit testleri geçiyor, viewer'da elle oynanıyor, `vmap` ile binlerce env'de SPS ölçüldü, aynı seed aynı sonucu veriyor.
**Anlamamız gerekenler:** Sabit shape + maske, `jnp.where` ile dallanmasız kod, auto-reset.

---

## Faz 2 — İlk öğrenme (≈2 hafta)

**Hedef:** Eğitim zincirinin uçtan uca çalıştığını görmek.
- IPPO, önce MLP, sonra 1D-CNN + GRU
- Görev: 1v0, rakip bayrağı al ve üsse getir
- Shaping ağırlıklarının ilk değerleri (`rewards.json`)
- Checkpoint (ağırlık, optimizer, adım, RNG, config, git commit), TensorBoard

**Kapı:** 1v0 görevde ≥%90 başarı (RS WP2 kriteri). Viewer'da AI izleme modu çalışıyor.
**Anlamamız gerekenler:** Recurrent PPO'da hidden-state saklama ve `done` maskeleme, `terminated` vs `truncated`.

---

## Faz 3 — Tam oyun + kural tabanlı botlar (≈3 hafta)

**Hedef:** Gerçek CTF, iletişimsiz.
- Lazer, respawn, saha yarısı. `game_rules.json`'daki `null` alanlar burada karara bağlanır.
- JAX'ta saf fonksiyon olarak heuristic botlar (saldırgan + savunmacı)
- MAPPO + CTDE critic, 2v2 botlara karşı

**Kapı:** Takımımız botları tutarlı şekilde yenmeye başlıyor.
**Anlamamız gerekenler:** CTDE, merkezi critic girdisi, çok ajanlı kredi ataması.

---

## Faz 4 — İletişim + erken H1 kontrolü (≈3 hafta) ⚠️ En kritik faz

**Hedef:** İletişimin işe yarayıp yaramadığını erken görmek.
- Mesaj başı + mean pooling. Takım, chunk boyunca birlikte açılır. Minibatch'ler env bazında bölünür.
- Kanal modları: open / closed / zeroed / shuffled
- Botlara karşı iletişimli vs iletişimsiz karşılaştırma

**Kapı:** İletişim açıkken anlamlı bir galibiyet farkı var. **Yoksa:** FOV'u daralt, görüşü kapatan duvarlar ekle, haritayı büyüt, sonra tekrar dene.
**Anlamamız gerekenler:** DIAL/CommNet/TarMAC farkları, mesaj üzerinden gradyan akışı.

---

## Faz 5 — Self-play + insan vs. bot (≈3–4 hafta)

**Hedef:** Botların ötesine geçen stratejiler.
- Rakip havuzu (%70 güncel, %30 geçmiş), GPU'da yığılmış parametreler
- Sabit değerlendirme seti (botlar + dondurulmuş checkpoint'ler) ve Elo
- Shaping'in sıfıra anneal edilmesi
- **İnsan vs. bot:** viewer'da bir ajanı insan klavyeyle kontrol eder, diğerleri politika. Simülasyon yine JAX olduğu için sim farkı oluşmaz.

**Kapı:** Elo istikrarlı artıyor, politika çökmüyor. İnsan oyuncuyla oynanabilir bir maç var.
**Anlamamız gerekenler:** Non-stationarity, cyclic forgetting, Elo.

---

## Faz 6 — Deneyler ve analiz (≈4–6 hafta)

**Hedef:** RS'deki hipotezleri (H1, H2) test etmek.
- 5 seed, `vmap` ile paralel. Koşullar: iletişim × FOV (120/180) × takım boyutu (2v2/3v3)
- Ablasyonlar (zeroed/shuffled), 3v3 → 5v5 genelleme
- Mesaj kaydı (oyun durumu etiketleriyle), PCA/t-SNE, K-means, silhouette
- Mann–Whitney U, bootstrap güven aralıkları

**Kapı:** RS WP4 kriterleri.

---

## Faz 7 — Görsel aktarım ve demo (Faz 5'ten itibaren paralel yürütülebilir)

- Maç kayıt formatı (`.npz`/JSON: kare başına poz, yönelim, bayrak, lazer olayları, mesajlar)
- Hedef motorda (Godot / Unity / Blender, **henüz karar verilmedi**) replay
- **Opsiyonel:** İnsanların güzel motorda oynaması isteniyorsa, motor yalnızca render ve girdi katmanı olur. Simülasyon JAX'ta kalır ve durum motora soket üzerinden canlı gönderilir. Böylece sim farkı yine oluşmaz.

---

## Açık noktalar (config'lerdeki `null`'lar)

- Lazer: sadece kendi sahanda mı vurulur, taşıyıcı ateş edebilir mi, menzil/cooldown, tek vuruş mu
- Kendi bayrağın çalınmışken skor yapılabilir mi
- Respawn koruması
- Bölüm süresi, hızlar, sensör menzili (viewer'da oynayarak ayarlanacak)
- Shaping ağırlıkları ve anneal süresi
- Işınlar üsleri görür mü / üslerde durur mu, vurulan ajanlar görünür mü
- Bayrak kaidesi üste mi, sahanın başka bir yerinde mi
- Görsel motor seçimi
