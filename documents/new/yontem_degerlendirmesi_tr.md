# DECO-SWARM v2 — Yöntem Değerlendirmesi ve Önerilen Yol Haritası

> Kapsam: `documents/01–06` (arkadaşının önerdiği mimari ve yol haritası) ile proje gereksinim belgesinin (PDF, RS) karşılaştırmalı değerlendirmesi, açık noktalar ve önerilen ilerleme yöntemi.
> Tarih: 2026-11-27

---

## 1. Kısa Cevap

- **Genel yön doğru.** Saf JAX, başsız (headless) simülasyon, sabit boyutlu diziler + maske, ego-merkezli gözlem, GRU + MAPPO + CTDE, 3 aşamalı müfredat ve havuzlu self-play, literatürle ve donanım kısıtıyla uyumlu. Temeli değiştirmeye gerek yok.
- **Asıl sorun sıralama.** Roadmap şelale (waterfall) gibi: önce simülasyonu tamamen bitir, sonra viewer, sonra ajan, sonra müfredat. RL'de ortam tasarımı hataları (eksik gözlem, ödül istismarı, oynanamaz kurallar) **ancak eğitim yapınca** ortaya çıkar. Simülasyonu baştan tamamen bitirmek yerine **ince ve uçtan uca çalışan bir dilim** (vertical slice) kurup her aşamada gerçekten bir şey eğitmeyi öneriyorum (Bölüm 4).
- **Dokümanlar büyük ölçüde v1 (tek ajanlı navigasyon) sisteminin çevirisi.** Ödüller, gözlem düzeni, müfredat eşikleri ve 5 kurallı checkpoint sistemi navigasyon için tasarlanmış. CTF için birkaç kritik noktada güncellenmesi gerekiyor (Bölüm 3).
- **"Görselleri güzelleştirmek için başka ortama aktarma" hedefi** şimdiden tasarımı etkiliyor: birimler, sensör sözleşmesi ve politika dışa aktarımı en baştan sabitlenmeli. En ucuz ve en sağlam yol, **JAX'ta oynatılan maçları kaydedip başka motorda (Godot/Unity/Blender) yeniden oynatmak**. Bu durumda canlı çıkarım (inference) gerekmez (Bölüm 5).

---

## 2. Önerilen Yöntemin Güçlü Yanları (Korunmalı)

| Konu | Neden doğru |
| :--- | :--- |
| Başsız simülasyon ↔ salt-okunur viewer ayrımı | Eğitim hızını korur. Aynı ayrım ileride başka motora aktarmayı da kolaylaştırır. |
| Fonksiyonel `reset/step` API, değişmez `EnvState` PyTree | JAX'ın `jit`/`vmap`/`scan` modeliyle birebir uyumlu (RS IR-API-01 ile aynı). |
| `MAX_SEGMENTS` padding + `segment_mask` | GPU'da değişken uzunluklu harita için doğru çözüm. |
| Global koordinat vermemek, ego-merkezli gözlem | Harita ezberini engeller. Kısmi gözlemlenebilirlik zaten araştırmanın konusu. |
| 60 Hz fizik / 10 Hz karar, raycast sadece makro-adım sonunda | Tünelleme engellenir ve raycast maliyeti 6 kat düşer. |
| `terminated` vs `truncated` ayrımı, GRU "pre-step hidden state" ve `(1-done)` maskeleme notları | Recurrent PPO'nun en yaygın iki hatası. Bunları baştan yazmış olmak değerli. |
| Tüm buffer üzerinde advantage normalizasyonu, gradyan kırpma, entropi | Standart ve doğru PPO ayarları. |
| Mann–Whitney U, 5 seed, kanal sıfırlama/karıştırma ablasyonları | Lowe et al. (2019) uyarısını ciddiye alan, sağlam bir deney tasarımı. |

---

## 3. Sorunlar, Tutarsızlıklar ve Eksikler

### 3.1 Gözlem tasarımı CTF için yetersiz (kritik)

- **Semantik sınıflar ayrışmıyor.** Doküman 02'deki ışın başına 7 değer `[dist, is_wall, is_ally, is_enemy, is_objective, cos, sin]`, bayrak ile üssü tek bir "objective" sınıfında birleştiriyor. RS ise `C = {wall, ally, enemy, flag, base, none}` diyor. CTF'de **kendi bayrağın / rakip bayrak** ve **kendi üssün / rakip üs** ayrımı olmadan ajan neye gittiğini bilemez.
  Öneri: `{none, wall, ally, enemy, own_flag, enemy_flag, own_base, enemy_base}` one-hot + "çarpılan ajan bayrak taşıyor mu" biti. Ayrıca RS'deki IR-DATA-02 (`K(1+|C|)`) formülü de buna göre güncellenmeli.
- **`rel_cos`, `rel_sin` sabit.** Bu değerler her ışın indeksi için hep aynı. Gözleme koymak rollout buffer'ını yaklaşık %28 büyütür ve bilgi taşımaz. 8 GB GPU'da bu fark hissedilir: 512 env × 6 ajan × 128 adım × 640 float ≈ 1 GB. Bunun yerine ağın içinde sabit bir pozisyon kodlaması olarak ekle.
- **Tek bir ego-polar yön var.** CTF'de en az iki referans gerekir: **kendi üssüne yön** (RS FR-SEN-03 bunu açıkça istiyor) ve **rakip bayrak kaidesine yön**. Kaide konumu sabit harita bilgisi olduğu için hile sayılmaz. Düşürülmüş bayrağın yeri ise verilmemeli: bunu iletişim taşımalı, araştırmanın ilginç kısmı da zaten bu.
- **Durum vektörü 2 değer.** Şunlar gerekli: bayrak taşıyor muyum, kendi bayrağımız çalındı mı, tag/respawn sayacı, kendi yarı sahamda mıyım.

### 3.2 İletişim kanalının nasıl öğrenileceği belirsiz (kritik)

RS "mesaj başı πθ'nin parçası, hareketle aynı ödülle şekillenir" diyor. Bunun iki çok farklı yorumu var:

1. **Mesaj = stokastik aksiyon (RIAL tarzı):** Mesaj örneklenir ve PPO ile pekiştirilir. Uygulaması kolay ama kredi ataması çok zor. Literatürde genelde yavaş öğrenir ya da hiç öğrenmez.
2. **Mesaj = türevlenebilir çıktı (DIAL/CommNet/TarMAC tarzı):** Gradyan, alıcının kaybından gönderenin ağına akar. **Önerim bu.**

JAX'ta türevlenebilir yaklaşımın pratik karşılığı:
- PPO güncellemesinde chunk (32 adım), **takımın tüm ajanlarıyla birlikte** `lax.scan` ile yeniden açılır. Carry, tüm ajanların GRU durumunu ve son mesajlarını içerir. Mesajlar tampondan okunmaz, güncel parametrelerle yeniden hesaplanır. Aksiyonlar ve log-prob'lar ise tampondan gelir.
- Bu nedenle **minibatch'ler ajan bazında değil, ortam (env) bazında** bölünmeli. Aynı takımın ajanları aynı minibatch'te olmalı. Doküman 04'teki "chunk'ları karıştır" adımı buna göre düzenlenmeli.
- Chunk başındaki mesaj tampondan alınır ve `stop_gradient` uygulanır (truncated BPTT).
- Bu karar Phase 3'ten önce verilmeli, çünkü buffer yapısını ve güncelleme döngüsünü belirliyor.

### 3.3 Oyun kuralları eksik tanımlı

- **Tag kuralı muğlak** ("proximity/LOS tagging"). Klasik CTF'de bir rakibi yalnızca **kendi yarı sahanda** tag'leyebilirsin. Bu kural yoksa oyun dejenere olur: herkes kaidede kamp kurar.
  Öneri: otomatik tag (ayrı bir aksiyon başı yok), yani kendi sahanda rakiple temas olunca rakip bayrağı düşürür ve N saniyelik respawn cezası alır. Aksiyon uzayı küçük kalır.
- **Düşen bayrak** ne olur? Yerinde mi kalır, kaideye mi döner, kendi takımı dokununca mı döner? Bu seçilmeli.
- **Bölüm sonu:** RS "ilk capture'da biter" diyor (FR-SIM-05). Bu basit ve iyi, korunmalı. 300 adım (30 s) ise 1000 px'lik haritada gidiş-dönüş ve çatışma için kısa olabilir. 400–600 adım ile test edilmeli.

### 3.4 Ödül ve müfredat navigasyondan kalma

- Doküman 01/03'teki ödüller (hedefe ilerleme, keşif ızgarası, zaman ölçekli hedef bonusu, `97.5%` terfi eşiği) navigasyon görevine göre ayarlanmış. CTF'de:
  - Takım ödülü sıfır toplamlı olmalı: capture `+1`, yenilgi `-1`, beraberlik `0`.
  - Şekillendirme ödülleri (bayrağa yaklaşma, pickup, taşıyıcıyı tag'leme) **eğitim boyunca sıfıra doğru azaltılmalı (anneal)**. Yoksa self-play'de ajanlar kazanmak yerine şekillendirme ödülünü toplamayı öğrenir.
  - Terfi eşikleri aşamaya göre ayrı belirlenmeli. Örneğin Aşama 1 için %90 navigasyon başarısı (RS WP2 kriteri), Aşama 2 için botlara karşı %80 galibiyet (RS WP3 kriteri).
- **Self-play'de "başarı oranı" tanımsız.** Rakip sürekli değiştiği için galibiyet oranı ilerlemeyi göstermez. İlerlemeyi **sabit bir değerlendirme setine karşı** ölçmek gerekir: heuristic botlar + dondurulmuş eski checkpoint'ler + Elo.

### 3.5 5 kurallı checkpoint sistemi aşırı karmaşık

- Rollback + LR yarılama, tekrarlanabilirliği (RS NFR-REP-01) zorlaştırır ve self-play'de tetikleme koşulu (başarı oranı) anlamsız hale gelir.
- Öneri: periyodik tam durum kaydı (ağırlık, optimizer, adım, RNG, config, git commit; RS IR-DATA-04), `latest`/`final` kayıtları ve self-play havuzu için periyodik snapshot yeterli. Rollback'i ancak gerçekten çöküş görürsen ekle.

### 3.6 Performans hedefi yanlış yerde

- **10 000 SPS JAX için çok düşük bir çıta.** Basit 2D ortamlar tek GPU'da kolayca 10⁶+ SPS'e ulaşır. Raycast maliyetine bakalım: 1024 env × 6 ajan × 90 ışın × ~50 primitif ≈ 2.8×10⁷ kesişim/adım. Bu GPU için önemsiz.
- **Darboğaz simülasyon değil, ağ eğitimi olacak** (CNN + GRU + chunked BPTT). Ölçülecek metrik "uçtan uca eğitim SPS'i" olmalı.
- **Deney bütçesi hesaplanmalı.** RS'deki koşullar: iletişim açık/kapalı × FOV 120°/180° × 2v2/3v3 × 5 seed = **40 eğitim koşusu**. Ayrıca iki laptop var. Bunun yolu **seed'leri `vmap` ile paralel eğitmek** (PureJaxRL tekniği): 5 seed tek süreçte, tek GPU'da aynı anda koşar.

### 3.7 Tekrarlanabilirlik detayı

- JAX GPU'da bazı işlemler (scatter-add vb.) deterministik değildir. FR-SIM-07 için `XLA_FLAGS=--xla_gpu_deterministic_ops=true` kullanılmalı ve bunun bir testi yazılmalı.

### 3.8 Tekerleği yeniden icat etme

- **JaxMARL** (RS'de [15] olarak zaten referans) içinde `ippo_rnn` / `mappo_rnn` uygulamaları var: recurrent MAPPO, GAE, done-maskeleme hazır. PPO döngüsünü sıfırdan yazmak yerine bunu **referans alıp uyarlamak** haftalar kazandırır ve hata riskini düşürür. Ortam arayüzünü de JaxMARL'ın `MultiAgentEnv` arayüzüne uygun tasarlamak iyi olur.
- Kütüphaneler: `flax`, `optax`, `orbax-checkpoint`, `distrax` (dağılımlar için).

### 3.9 Self-play ve botların JAX'a özgü gereksinimleri

- **Heuristic botlar da `jit` uyumlu saf fonksiyonlar olmalı.** Global durumu (privileged state) kullanabilirler. Python tarafında yazılırsa tüm hız kaybolur.
- **Rakip havuzu** GPU'da sabit boyutlu, yığılmış (stacked) bir parametre PyTree'si olarak tutulmalı (ör. K=8 slot). Her env bir indeksle rakibini seçer, rakip ağı `vmap` ile o parametrelerle çalışır.

### 3.10 H1 (%20 galibiyet artışı) riski

- İletişimin gerçekten fark yaratması için **ortamın iletişimi gerektirmesi** lazım. Bunun için dar FOV (120°), görüşü kapatan duvarlar, savunmacının göremediği yönlerden gelen saldırılar ve düşen bayrağın yerinin yalnızca gören ajan tarafından bilinmesi gibi özellikler gerekir.
- Bu hipotez **mümkün olan en erken aşamada** test edilmeli: botlara karşı, self-play'den önce (Bölüm 4, Adım 4). Eğer iletişim botlara karşı fark yaratmıyorsa haritalar ve FOV erkenden ayarlanabilir.

---

## 4. "Simülasyon Önce Tamamen Hazır mı Olmalı?" — Önerim: Hayır

### Sabitlenmesi gerekenler vs. evrilebilecekler

| Baştan sabitle (sözleşme) | Yol boyunca evrilebilir |
| :--- | :--- |
| `reset/step` API imzası ve shape'ler | Fizik detayları (sürtünme, ivme) |
| Gözlem ve aksiyon şeması (ışın sayısı, sınıflar, birimler) | Harita çeşitliliği, prosedürel üretim |
| Birim sistemi (metre/saniye) | Ödül ağırlıkları |
| Temel CTF kuralları (pickup, capture, tag, bitiş) | Viewer'ın görsel cilası |

### Önerilen sıralama (her adımın sonunda bir şey **eğitilir**)

| Adım | İçerik | Bitti sayılma kriteri | Tahmini süre |
| :--- | :--- | :--- | :--- |
| **0. Sözleşme** | Repo iskeleti, config sınıfları, `SIM_CONTRACT.md` (birimler, gözlem/aksiyon şeması, kurallar) | Sözleşme dokümanı + boş API'nin testleri | 1 hafta |
| **1. Minimal sim + viewer** | Açık arena + birkaç kutu, 1v1/2v2, hareket + kayan çarpışma, semantik raycast, bayrak pickup/capture (tag yok). Basit pygame viewer + insan modu | Unit testler, viewer'da elle oynanabiliyor, SPS benchmark | 2 hafta |
| **2. İlk öğrenme** | JaxMARL tabanlı IPPO, önce MLP sonra GRU. Görev: 1v0, rakip bayrağı al ve üsse getir | ≥%90 başarı (RS WP2 kriteri) | 2 hafta |
| **3. Tam CTF + botlar** | Tag, saha yarısı kuralı, respawn, düşen bayrak. JAX'ta heuristic saldırgan/savunmacı botlar. CTDE MAPPO, iletişimsiz | Takım botları yenmeye başlıyor | 3 hafta |
| **4. İletişim + erken H1 testi** | Türevlenebilir mesaj başı, attention/mean pooling, env bazlı minibatch. Botlara karşı iletişim açık vs kapalı | İletişim açıkken anlamlı bir fark var. Yoksa harita/FOV ayarla | 3 hafta |
| **5. Self-play** | Rakip havuzu (%70 güncel, %30 geçmiş), Elo, sabit değerlendirme seti, şekillendirme anneal | Elo monoton artıyor, çöküş yok | 3–4 hafta |
| **6. Deneyler ve analiz** | 5 seed × tüm koşullar (`vmap` ile), ablasyonlar (sıfırlama/karıştırma), 3v3→5v5, mesaj kaydı, PCA/t-SNE/K-means/silhouette | RS WP4 kriterleri | 4–6 hafta |
| **7. Görsel aktarım** | Maç kaydı → başka motorda replay (Bölüm 5) | Demo videosu | paralel yürütülebilir |

Bu sıralama RS'deki WP1–WP4 takvimiyle uyumlu, sadece işleri daha erken entegre ediyor. Avantajları:
- Gözlemde eksik bilgi, ödül istismarı, kural hatası gibi sorunlar **2. ve 3. haftalarda** fark edilir, 4. ayda değil.
- En büyük bilimsel risk olan "iletişim işe yarıyor mu" sorusu self-play'den **önce** test edilir.
- Her adımda çalışan bir demo olur. Bu, ara raporlar ve danışman görüşmeleri için de iyi.

---

## 5. Başka Ortama Aktarım (Görselleştirme) Stratejisi

### Seçenek A — Kayıt + replay (önerilen)
- JAX simülasyonu maçı oynar. Her karede ajan pozisyonu, yönelimi, bayrak durumu, tag olayları ve mesajlar `.npz`/JSON olarak kaydedilir. RS IR-DATA-03 zaten bu log'u istiyor, yani ekstra iş az.
- Godot/Unity/Blender bu kaydı **yalnızca görsel olarak yeniden oynatır**. 3D modeller, animasyon, ışıklandırma tamamen serbest.
- **Simülasyon uyuşmazlığı riski sıfır**, çünkü fizik yalnızca JAX'ta çalışır.

### Seçenek B — Hedef motorda canlı çıkarım
- Ağırlıklar ONNX'e veya düz NumPy'a aktarılır ve motor içinde politika çalıştırılır.
- Sensör ve fiziğin **birebir** kopyalanması gerekir: ışın sayısı/FOV/menzil, sınıf kodlaması, hız limitleri, 10 Hz karar frekansı, aksiyon anlamları. Küçük bir fark bile politikayı bozabilir (sim-to-sim gap).
- Etkileşim gerekiyorsa (ör. insan oyuncu ajanlara karşı oynayacaksa) mantıklı. Sadece güzel görüntü için gereksiz efor.

### Her iki seçenek için şimdiden yapılması gerekenler
- **Piksel yerine metre kullan.** Örneğin ajan yarıçapı 0.5 m, harita 50×50 m. Oyun motorları metrik çalışır, `px` birimleri aktarımda dönüşüm hatası kaynağı olur. Ekran pikseline dönüşüm yalnızca viewer'ın kamerasında yapılsın.
- Gözlem/aksiyon şemasını `SIM_CONTRACT.md` içinde sürümlü tut. Değişirse sürüm numarasını artır.
- Checkpoint'e config ve sözleşme sürümünü de yaz.

---

## 6. Mimari İçin Küçük Öneriler

- **Ağ boyutları makul.** 1D-CNN → 128, ego MLP → 32, GRU 128. Mesaj boyutu `d` için 8–16 ile başla. Küçük `d`, kümelenme analizini (H2) de kolaylaştırır.
- Işınlar dairesel değil (180° yay), dolayısıyla CNN'de dairesel padding gerekmez. 360° FOV denenirse `wrap` padding kullanılmalı.
- **CTDE critic girdisi:** MAPPO makalesindeki "agent-specific global state" (global durum + ajanın kendi özellikleri) iyi bir başlangıç. 5v5 genelleme testinde critic kullanılmadığı için critic'in takım boyutundan bağımsız olması şart değil.
- **"Kanal kapalı" baseline'ı:** Aynı mimari, mesaj girdisi her zaman sıfır, iletişim başı ise eğitimde gradyan almıyor. Böylece parametre sayısı aynı kalır ve karşılaştırma adil olur.
- **Aksiyonlar:** Hareket (5) + dönüş (3) multi-discrete yeterli. Tag otomatik olduğu için ek başa gerek yok.

---

## 7. Karar Verilmesi Gereken Açık Sorular

1. Mesaj öğrenimi: türevlenebilir (önerilen) mi, stokastik aksiyon mu?
2. Tag kuralı: kendi saha yarısında temasla otomatik tag (önerilen) mi, ayrı bir aksiyon mu?
3. Düşen bayrak: yerinde mi kalır, kaideye mi döner?
4. Görselleştirme aktarımı: replay (önerilen) mi, canlı çıkarım mı? Hedef motor hangisi?
5. PPO altyapısı: JaxMARL uyarlaması (önerilen) mı, sıfırdan mı?
6. Birimler: metreye geçiş (önerilen) onaylanıyor mu?

---

## 8. Özet

Arkadaşının mimarisi sağlam bir temel. Değiştirilmesini önerdiğim şeyler temelde değil, **sıralamada** ve **CTF'ye özgü detaylarda**:

1. Şelale yerine her adımda eğitim yapılan dikey dilimler.
2. Gözlemi CTF'ye uyarla: sınıfları ayır, iki referans yönü ekle, durumu genişlet, sabit cos/sin'i çıkar.
3. İletişimi türevlenebilir yap ve buna göre env bazlı minibatch kullan.
4. Tag/saha kuralını netleştir, ödülleri sıfır toplamlı + anneal edilen şekillendirme olarak kur.
5. Checkpoint sistemini sadeleştir, ilerlemeyi sabit değerlendirme setine karşı ölç.
6. Darboğazın eğitim olduğunu kabul et: seed'leri `vmap`'le, JaxMARL'ı referans al.
7. Görsel aktarım için metre birimleri + kayıt/replay.
