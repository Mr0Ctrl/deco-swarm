### DECO-SWARM: 28 Haftalık Geliştirme ve Faz Takvimi

#### FAZ 1: Temel Atma ve "Proof of Concept" (Hafta 1 - 4)

_Acil Hedef: RL ortamının mantığını kavramak ve tek ajanın (1v0) kör noktalara takılmadan hedefe gitmesini sağlamak._

- **Hafta 1-2: 1v0 Labirent Ortamı ve İlk Eğitim (Python)**
- Ortamın saf Python ve `Gymnasium` standartlarında kodlanması.

- Sürekli kinematik (Continuous Kinematics) fizik kurallarının ve Raycast matematiğinin Python'da yazılması.

- Ödül fonksiyonunun (hedefe yaklaşma, duvara çarpma cezası) tasarlanması.

- GRU (hafıza) destekli temel bir modelin PPO ile eğitilmeye başlanması.

- **Hafta 3-4: Aksiyon Uzayı ve Statik Çatışma (1v1 Statik)**
- Modelin labirenti çözdüğünün doğrulanması ve yerel minimumlardan (kendi etrafında dönme) kurtarılması.

- Çoklu-Ayrık (Multi-Discrete) aksiyon uzayının (hareket + dönüş + ateş) entegrasyonu.

- Haritaya hareketsiz hedefler eklenerek ajana "Raycast ile nişan al ve ateş et" refleksinin öğretilmesi.

#### FAZ 2: Performans Optimizasyonu ve Çoklu Ajan Geçişi (Hafta 5 - 9)

_Hedef: Python'un işlemci darboğazını C++ ile aşmak ve PettingZoo API'sine geçiş yapmak._

- **Hafta 5-7: C++ Köprüsü (pybind11)**
- Giderek yavaşlayan Python Raycast ve çarpışma (collision) algoritmalarının C++ tarafına yazılması.

- `pybind11` ve NumPy Zero-Copy (paylaşımlı bellek) köprüsü kurularak simülasyonun saniyede 10.000+ adıma hızlandırılması.

- **Hafta 8-9: MARL Ortamına Geçiş (PettingZoo)**
- Ortamın tek ajanlı `Gymnasium` yapısından, çok ajanlı `PettingZoo` yapısına dönüştürülmesi.

- CTDE (Merkezi Eleştirmen, Dağıtık Aktör) mimarisi için algı ve durum (observation/state) tensörlerinin ayrılması.

#### FAZ 3: Taktiksel Çatışma ve Ara Sunum (Hafta 10 - 14)

_Hedef: 14. haftadaki jüri sunumuna, kısıtlı bir takım çalışmasını başaran hızlı bir simülasyon altyapısıyla çıkmak._

- **Hafta 10-12: Dinamik 1v1 ve İletişim Temelleri**
- T ve CT ajanlarının birbirine karşı hareketli çatışma (Dinamik 1v1) eğitimi.

- Ajanların kendi aralarında "Tespit Edilen Hedef Koordinatını" paylaşacağı telsiz kanalının girdi vektörüne eklenmesi.

- **Hafta 13-14: 3v3 Sürü Prototipi ve 1. Dönem Sunumu**
- Takım havuzlama (Pooling / Permutation Invariance) katmanının çalıştırılması.

- **Ara Sunum Çıktısı (Midterm):** C++ ile hızlandırılmış, 1v0 labirenti kusursuz çözen, 1v1'de mantıklı ateş eden ve 3v3 formatında ilk ortak hareket denemelerini (gelişme aşamasında) sergileyen bir altyapı demosu.

---

#### FAZ 4: İleri Seviye MARL ve Self-Play (Hafta 15 - 20)

_Hedef: Ajanların "askeri manga" davranışlarını (çapraz ateş, pusu) kendiliğinden keşfetmesi._

- **Hafta 15-17: Karşıt Eğitim (Self-Play / League Training)**
- T ve CT modellerinin sürekli kendi eski versiyonlarına karşı maç yaparak taktik geliştirmesi (aşırı uyumu/ezberlemeyi önleme).

- Ödül şekillendirme (Reward Shaping): Bencil oynamayı cezalandırma, dost ateşi (friendly fire) kısıtları, takım için vurulmayı göze alma ödülleri.

- **Hafta 18-20: Heterojen Roller ve İnce Ayar (Fine-Tuning)**
- Silah çeşitlerinin (Tüfek, Pompalı, Keskin Nişancı) sisteme one-hot vektörlerle eklenmesi.

- Öğrenme oranını (Learning Rate) düşürerek eski refleksleri kaybetmeden yeni silah kurallarına adaptasyon (Fine-Tuning).

#### FAZ 5: İnsan-Yapay Zeka Etkileşimi ve Görselleştirme (Hafta 21 - 25)

_Hedef: Çıkan sonucu herkesin anlayabileceği, oynanabilir ve görsel bir formata sokmak._

- **Hafta 21-23: C++ Görselleştirme (Raylib / SFML)**
- Eğitim modundan (headless) çıkarak C++ içerisinden 60 FPS çalışan bir grafik arayüz (UI) ayağa kaldırma.

- Raycast konilerinin, mermilerin ve ajan can barlarının ekrana çizdirilmesi.

- **Hafta 24-25: Etkileşim Soketi (Socket/Human Client)**
- Klavye/fare komutlarının C++ motoruna aktarılması.

- Eğitilmiş yapay zeka sürüsünün, kontrolü insan tarafından sağlanan bir ajana karşı (veya onunla birlikte) test edilmesi.

#### FAZ 6: Doğrulama ve Final Raporu (Hafta 26 - 28)

_Hedef: Sistemin askeri doktrinlere uygunluğunu sayısal metriklerle kanıtlamak ve projeyi teslim etmek._

- **Hafta 26-27: Veri Analizi ve Karşılaştırma**
- Dağıtık modelin (telsizle haberleşen) başarı oranının, tüm haritayı gören merkezi bir modelle (Oracle Baseline) karşılaştırılarak istatistiklerinin çıkarılması.
- Sürü içi kopma mesafesi, mermi isabet oranı gibi metriklerin grafikleştirilmesi.

- **Hafta 28: Proje Teslimi**
- Yazılı raporun ve sunum slaytlarının tamamlanması. Final savunması.
