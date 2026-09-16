## 1. Projenin Amacı ve Kapsamı

Bu projenin temel hedefi, standart bir oyuncu laptopu donanımında eğitilebilen; basitleştirilmiş 2D taktiksel bir nişancı (shooter) arenasında merkeziyetsiz sürü davranışı sergileyen Çok Etmenli Pekiştirmeli Öğrenme (MARL) sistemi geliştirmektir. Ajanların (İHA/SİHA veya otonom piyade simülasyonu) haritada birbirinden bağımsız rastgele hareket etmesi yerine; aralarında kısıtlı bilgi paylaşarak çapraz ateş, pusu, odaya taktiksel giriş ve arka kollama gibi askeri manga disiplinlerini kendi kendilerine (emergent behavior) keşfetmeleri amaçlanmaktadır. Proje 28 haftalık bir geliştirme takvimine sahiptir ve nihai hedef tam teşekküllü 3v3 / 5v5 rekabetçi görevlerin (bomba kurma/çözme, bölge savunması) icra edilmesidir.

## 2. Teknoloji Yığını ve Sistem Mimarisi (Hibrit Yaklaşım)

Sistem, eğitim hızını maksimize etmek ve aynı zamanda görsel hata ayıklama (debug) yapabilmek için C++ ve Python dillerinin hibrit kullanımıyla tasarlanmıştır.

- **Simülasyon Çekirdeği (C++):** Harita geometrisi, hareket fiziği, raycast (ışın çarpışma) hesaplamaları ve mermi tünelleme gibi yüksek CPU tüketen döngüler saf C++ ile yazılacaktır.

- **Makine Öğrenmesi (Python/PyTorch):** Model mimarisi, ödül hesaplamaları, MAPPO algoritması ve ağın eğitimi Python üzerinde gerçekleşecektir.

- **Köprü (Pybind11 & Zero-Copy):** C++ çekirdeği ile Python arasındaki veri alışverişi, bellek kopyalama (deep-copy) darboğazını aşmak için paylaşımlı NumPy tampon belleği (Zero-copy) üzerinden yapılacaktır.

- **Ortam Standardı:** Python tarafındaki eğitim arayüzü, yaygın RL kütüphaneleriyle tam uyumluluk sağlamak için `PettingZoo` API standartlarında (wrapper) tasarlanacaktır.

- **Görselleştirme:** Simülasyon eğitim sırasında ışıksız (headless) çalışarak saniyede on binlerce adım atacak; ancak istendiğinde C++ içine gömülü hafif bir kütüphane (Raylib/SFML) ile 60 FPS grafik penceresi açılarak ajanların davranışları veya insan-AI rekabeti izlenebilecektir.

## 3. Simülasyon ve Fizik Kuralları

- **Fizik Modeli:** Ayrık ızgara (discrete grid) yerine, ajanların serbest koordinatlarda ($x, y$) gezindiği Sürekli Kinematik (Continuous Kinematics) yapısı kullanılacaktır.

- **Harita Geometrisi:** Duvarlar ve siperler sadece eksene paralel kutular (AABB) değil, çapraz ve açılı yapıları da desteklemesi için Serbest Çizgi Segmentleri (Line Segments) veya OBB olarak modellenecektir.
- **Zaman ve Karar Frekansı:** Mermi tünellemesini (bullet tunneling) engellemek ve akıcı bir oyun fiziği elde etmek için C++ motoru 60 Hz (saniyede 60 adım) hızında fizik hesaplayacaktır. Ancak yapay zeka ajanları her 6 karede bir (10 Hz frekansla) karar verecektir (Action Repeat). "Hareket" aksiyonları 6 kare boyunca uygulanırken, "Ateş Etme" eylemi silahın bekleme (cooldown) süresine tabi olacaktır.

## 4. Gözlem Uzayı (Observation Space) ve Algı

Ajanlar dünyanın mutlak koordinatlarını bilmez; tüm girdiler ajanın gövdesini merkez alan (ego-centric) bağıl bir formatta verilir. İleride eklenebilecek sensörler için tensörün sonunda "Zero-Padding" (boş sıfırlar) bırakılacaktır.

- **Öz Bilgi:** Can, mermi durumu ve elindeki silah tipi (one-hot encoding).

- **Raycast (Gözler):** Ajanın baktığı yönde 180 derecelik açıyı tarayan sabit sayıda ışın. Her ışın "Mesafe" ve "Çarpan Nesne Tipi" döndürür.

- **Pusula / Hedef Vektörü:** O an atanmış olan taktiksel hedefin (örneğin A bomba alanı veya girilecek odanın kapısı) ajana göre bağıl açısı ve mesafesi ($\Delta x, \Delta y, \theta$).

- **Hafıza (GRU/LSTM):** Ajanın kısıtlı görüş alanından çıkan düşmanları veya kör noktaları unutmaması, ayrıca çıkmaz sokaklarda kendi etrafında dönmemesi (spinning behavior) için model mimarisinin içine Tekrarlayan Sinir Ağı (GRU hücresi) entegre edilecektir. Frame stacking kullanılmayacaktır.

## 5. Aksiyon Uzayı (Action Space)

Oyun kontrol hissini (klavye/fare) en iyi yansıtan Çoklu-Ayrık (Multi-Discrete) yapı kullanılacaktır. Ajan her adımda birbirinden bağımsız alt kararlar alır:

- _Yönelim:_ Dur, İleri, Geri, Sola Kay, Sağa Kay.
- _Dönüş:_ Sabit, Sola Dön, Sağa Dön.
- _Tetik:_ Ateş Etme, Ateş Et.

## 6. Haberleşme ve Karar Mekanizması

- **Eğitim Mimarisi (CTDE):** Eğitim sırasında (Centralized Training) tüm haritayı gören merkezi bir eleştirmen (Critic) bulunacak; sahada ise (Decentralized Execution) ajanlar yalnızca kendi algılarıyla hareket edecektir.

- **Haberleşme Kısıtı:** Ajanlar serbest gürültülü float vektörleri ile değil, taktiksel anlamı baştan tanımlanmış "Tespit Edilen Hedef Koordinatı" gibi somut verileri gerçek zamanlı telsiz kanalı üzerinden birbirleriyle paylaşacaktır.

## 7. Eğitim Stratejisi

Ajanların donanımı kilitlemeden ve seyrek ödül (sparse reward) tuzağına düşmeden eğitilmesi için Aşamalı Öğrenme (Curriculum Learning) uygulanacaktır:

1. **Faz 1 (1v0 Navigasyon):** Düşman yok. Ajan sadece hedefe raycast ve GRU kullanarak duvarlara çarpmadan gitmeyi öğrenir.

2. **Faz 2 (Statik Atış):** Hareketsiz ve tehlikesiz hedeflere raycast ile nişan alıp mermi israf etmeden ateş etme.

3. **Faz 3 (1v1 Dinamik):** Teke tek siper alma ve hareketli çatışma.

4. **Faz 4 (3v3 / 5v5 Sürü):** Takım arkadaşları sahaya eklenir, kısıtlı haberleşme açılır, MAPPO ve Self-Play (lig usulü kendi kendine oynama) devreye girer.
