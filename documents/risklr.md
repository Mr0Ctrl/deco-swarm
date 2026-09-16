**Sıfırdan Eğitim Gerektiren Değişiklikler (Hard Reset Riskleri)**

- **Gözlem Uzayı (Observation Space) Boyutunun Değişmesi:** Ağın ilk katmanının ağırlık matrisi girdi vektörünün boyutuna doğrudan bağlıdır. İleride "Raycast sayısını 16'dan 24'e çıkaralım", "Görüş açısını 180°'den 270°'ye genişletelim" veya "Vektöre pusula bilgisi ekleyelim" derseniz, eski model ağırlıkları yeni tensör boyutlarıyla uyuşmaz ve model sıfırdan eğitilmek zorunda kalır. (Önlem: Baştan sıfır dolgulu (zero-padding) yedek kanallar bırakmak).

- **Aksiyon Uzayının (Action Space) Değişmesi:** Seçtiğimiz Çoklu-Ayrık (Multi-Discrete) yapıda daha sonra yeni bir eylem (örneğin "siper al", "silah değiştir", "el bombası at") eklemek veya eksen ayrımını değiştirmek aktör ağının çıkış katmanını bozar.

- **Karar Frekansı / Tick Rate Değişimi:** 10 Hz karar frekansıyla eğitilen bir modeli 60 Hz'e çekmek basit bir fine-tuning ile yapılamaz. Zaman adımı ($\Delta t$) değiştiğinde gelecekteki ödüllerin indirgenme katsayısı ($\gamma^{\Delta t}$) ve eylemlerin çevreye etki süresi tamamen değişir; model ya aşırı reaktif davranıp titremeye (chattering) başlar ya da öğrenilmiş reflekslerini kaybeder.

- **Ödül Skalasının ve İşaretlerinin Değişmesi:** Eğitim ortasında ceza veya ödüllerin ağırlıklarını radikal biçimde değiştirmek (örneğin galibiyet ödülünü 10'dan 100'e çekmek, duvara çarpma cezasını tersine çevirmek) PPO'nun değer fonksiyonunu (Critic / Value Loss) patlatır ve politika anında çöker.

**Modelin Öğrenmesini Engelleyen Taktiksel Tuzaklar (Pathologies)**

- **Ödül İstismarı (Reward Hacking) ve Pasiflik:** Ajanlar sizin amacınızı değil, yazdığınız matematiksel skoru maksimize eder. Ölüm cezası fazla verilirse ajanlar doğuş noktasından çıkmayıp köşelerde saklanır; mermi israfı cezası yüksek olursa hiç ateş etmez; hedefe yaklaşma ödülü Öklid mesafesiyle verilirse duvar arkasında hedefe en yakın noktaya yüzünü yapıştırıp orada kilitlenir.

- **Kısmi Gözlemlenebilirlik (POMDP) ve Kendi Etrafında Dönme (Spinning):** 180° kısıtlı görüşte ajan hedefi görüş alanından çıkardığı an hedefin nerede olduğunu unutursa, hedef açısını tekrar yakalayabilmek için olduğu yerde sürekli fırıldak gibi dönme davranışına (spinning behavior) takılır. Bunu aşmak için iç hafıza (GRU/LSTM) ve bağıl yön vektörü şarttır.

- **Seyrek Ödül (Sparse Reward) Çıkmazı:** Eğer ajana yalnızca "raundu kazandığında +10" verirseniz, haritada rastgele hareket eden acemi bir ajanın tesadüfen koridorları geçip, düşmanı bulup, vurup kazanma ihtimali pratikte sıfırdır. Bu durum modelin ilk 5 milyon adım boyunca hiçbir şey öğrenememesine yol açar.

**Çoklu Ajan (MARL) ve İletişim Riskleri**

- **Haberleşme İhmali (Communication Collapse):** Ajanlara serbest bir telsiz vektörü verildiğinde, eğitimin ilk aşamalarında bu kanaldan sadece rastgele gürültü akar. Ajanların sinir ağları bu gürültüden kaçınmak için iletişim girdisinin ağırlıklarını hızla sıfıra indirger. Kanal kuralcı veya belirgin bir ödülle desteklenmezse ajanlar birbirini tamamen yok saymayı öğrenir.
- **Ajan Sıralaması Karmaşası (Permutation Problem):** Takım arkadaşlarının koordinatları düz bir liste (`[arkadaş1, arkadaş2]`) olarak ağa verilirse, 1. arkadaş öldüğünde veya liste sırası kaydığında model kimin nerede olduğunu ayırt edemez ve takım bilinci çöker. (Önlem: MaxPooling veya Deep Sets mimarisi kullanmak).

- **Kredi Atama (Credit Assignment) Çıkmazı:** 3v3 bir maçta bir ajan harika bir çapraz ateş açısı tuttuğu halde, başka bir takım arkadaşı arkadan vurulup maçı kaybettirirse, standart algoritmalar doğru pozisyon alan ajanı da cezalandırır. Bu dengesizlik ajanların bencil oynamasına neden olur.

- **Taş-Kağıt-Makas Kısır Döngüsü ve Unutma (Catastrophic Forgetting):** Self-play liginde T takımı belirli bir A hücumu geliştirir, CT takımı bunu ezip geçer. CT yeni savunmayı öğrenirken önceki temel reflekslerini unutabilir (aşırı uyum). Modeller birbirinin açığını kapattıkça döngüsel bir taktik kısırlığı oluşabilir.

**Donanım ve Kodlama Darboğazları**

- **Pybind11 Bellek Kopyalama (Deep-Copy) Tuzağı:** C++ motoru ne kadar hızlı olursa olsun, her adımda gözlem tensörleri C++'tan Python'a bellekte kopyalanarak aktarılırsa simülasyon hızı yerlerde sürünür. Paylaşımlı bellek (Zero-copy NumPy buffer) zorunludur.

- **Sürekli Kinematik Raycast Yükü:** 2D sürekli fizik motorunda her adımda 10 ajan için 16'şar raycast hesaplamak, saniyede yüzbinlerce çizgi-kutu kesişim testi demektir. Çarpışma algoritmaları uzamsal ızgaralarla (spatial hashing / grid partitioning) optimize edilmezse CPU tıkanır.
