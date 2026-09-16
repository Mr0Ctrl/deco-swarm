```

deco-swarm/                         # Repo Kökü (Git)
├── documents/                      # Dokümantasyon
│   ├── akış.md
│   ├── geliştime.md
│   ├── ptoje_tanımı.md
│   └── risklr.md
│
├── configs/                        # Fizik ve Eğitim Konfigürasyonları
│   ├── physics.yaml                # Ray sayısı, FOV, hızlar, harita boyutları
│   └── train_1v0.yaml
│
├── src/                            # Ana Paket
│   ├── __init__.py
│   │
│   ├── engine/                 # SAF FİZİK MOTORU (Pygame YOK, Headless)
│   │   ├── __init__.py
│   │   ├── world.py            # Tüm ajanları, mermileri, duvarları tutan ana simülasyon
│   │   ├── agent.py            # Ajan kinematiği (konum, yönelim, can)
│   │   ├── map.py              # Duvar segmentleri ve engeller
│   │   └── raycast.py          # Çizgi-ışın kesişim matematiği
│   │
│   ├── rendering/              # GÖRSELLEŞTİRME KATMANI (Pygame burada)
│   │   ├── __init__.py
│   │   └── pygame_renderer.py  # World nesnesini alıp ekrana 60 FPS çizen pencere
│   │
│   ├── envs/                   # GYMNASIUM / RL ARAYÜZÜ
│   │   ├── __init__.py
│   │   └── single_agent_env.py # Headless ve Human render_mode desteği burada birleşir
│   │
│   ├── models/                 # PyTorch Modelleri (GRU, Actor-Critic)
│   │   └── actor_critic.py
│   │
│   └── training/               # PPO Eğitim Betikleri
│       └── ppo.py
│
├── cpp/                            # C++ Hızlandırma Katmanı (Faz 2)
│   ├── CMakeLists.txt
│   └── ...
│
├── play.py                         # MERKEZİ GÖRSEL ÇALIŞTIRICI (Head Modu)
│                                   # --mode test     : Haritayı ve ray'leri klavyeyle test et
│                                   # --mode watch    : Eğitilmiş modeli canlı izle
│                                   # --mode human-vs-ai : İnsan botlara karşı
│
├── train.py                        # MERKEZİ EĞİTİM BETİĞİ (Headless Modu)
│                                   # Arka planda penceresiz son sürat eğitir
│
├── .gitignore
├── pyproject.toml
└── README.md

```
