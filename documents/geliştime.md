**DECO-SWARM: Geliştirme Süreci ve İş Akışı Raporu**

Aşağıda, sistemin kodlanma aşamasından nihai modele kadar izlenecek yolu, teknoloji yığınını ve performans hedeflerini özetleyen geliştirme süreci raporu yer almaktadır.

### 1. Teknoloji Yığını ve Framework Seçimleri

- **Derin Öğrenme (Model ve Eğitim):** PyTorch. Esnek yapısı ve modüler ağ tasarımına (örneğin çevre algısı ile telsiz verisini ayırma) tam uyum sağlaması nedeniyle temel framework olacaktır.

- **MARL Algoritma Altyapısı:** Sıfırdan PPO matematiği yazmak yerine, endüstri standardı olan algoritmaları barındıran **CleanRL** veya **Ray RLlib** kullanılacaktır.

- **Ortam Arayüzü (API):** Simülasyonun algoritmalara bağlanabilmesi için standartlaştırılmış **PettingZoo** (çoklu ajan) ve **Gymnasium** (tekli ajan) yapıları (wrapper) kullanılacaktır.

- **C++ Köprüsü ve Görselleştirme:** C++ ile Python arasındaki veri aktarımı (sıfır kopyalama/zero-copy memory) için **pybind11**; debug ve izleme penceresi için **Raylib** tercih edilecektir.

### 2. Modüler Model Tasarımı ve Öğrenme Odakları

Modelin girdilerini anlamsal özelliklerine göre parçalı (multi-modal) işlemek, ağın öğrenme kalitesini artırır. Model tasarımı için şu mimari konseptlerin araştırılması gereklidir:

- **Özellik Çıkarıcılar (Feature Extractors):** Raycast (çevre) girdisinin ayrı bir sinir ağı bloğundan (örneğin 1D-CNN veya MLP), iletişim ve takım durumunun (Comm Vector) ayrı bir bloktan geçirilip daha sonra birleştirilmesi (Late Fusion).
- **POMDP ve Zaman Serileri:** Ajanların kısmi gözlemlenebilirlik (POMDP) altında kör noktalara takılmaması için PyTorch'ta **GRU/LSTM** (Tekrarlayan Sinir Ağları) entegrasyonu.

- **Kredi Atama (Credit Assignment):** Takım arkadaşlarının başarısına göre ajanların nasıl ödüllendirileceğini belirleyen algoritmik yapılar.

### 3. Aşamalı Eğitim (Curriculum) ve Harcanabilir Modeller

RL geliştirme süreci oldukça deneyseldir. Eğitim stratejisi "harcanabilir" prototipler üzerine kurulacaktır.

- **Adım 1 (Isınma / Çöpe Atılabilir Model):** Hiçbir düşmanın veya takım arkadaşının olmadığı, yalnızca 1v0 labirentte hedefe gitmeyi öğrenen basit bir "proof of concept" modeli eğitilecektir. Bu süreç tamamen RL hiperparametrelerini (learning rate, entropy) kavramak içindir.

- **Adım 2 (Genişletme):** Model 1v1 çatışma ortamına taşındığında, ortam karmaşıklaşacağı için sıfırdan yeni bir yapıya geçilecektir. İlk aşamanın veri yapısı (sıfır dolgusu/zero-padding bırakılarak) dinamik tutulacaktır.

### 4. Performans Hedefleri ve Python'dan C++'a Geçiş Akışı

Eğitim süresini laptop üzerinde makul (gecelik) seviyelerde tutmak için simülasyonun hedefi saniyede en az **10,000+ adım** (step/sec) üretmek olmalıdır. Bu hedefe ulaşmak için kademeli bir geçiş uygulanacaktır:

- **Faz A (Saf Python Prototipi):** İlk 1v0 labirent denemeleri NumPy ile yazılmış saf Python ortamında yapılacaktır. Mantık ve ödül fonksiyonu burada doğrulanır.

- **Faz B (Darboğazların Taşınması):** Ortam 3v3 seviyesine geldiğinde Python CPU'yu yormaya başlayacaktır. Yalnızca **Raycast kesişim matematiği** ve **çarpışma algılama (collision detection)** C++ ile yazılarak pybind11 üzerinden Python'a bağlanacaktır.

- **Faz C (Motorun Tamamen Taşınması):** Nihai performansa ihtiyaç duyulduğunda, ajanların hareket fiziği ve oyun kuralları tamamen C++'a aktarılır. Python tarafı yalnızca GPU/CPU üzerinde PyTorch ile ağırlık güncelleyen bir denetleyici (controller) olarak kalır.
