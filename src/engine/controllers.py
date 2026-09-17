# src/deco_swarm/engine/controllers.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import pygame


@dataclass
class AgentAction:
    """Ajanın bir adımda yapabileceği eylemler paketi."""
    move_fwd: float = 0.0      # +1.0 İleri, -1.0 Geri, 0.0 Dur
    strafe: float = 0.0        # +1.0 Sağa, -1.0 Sola, 0.0
    rotate_dir: float = 0.0    # +1.0 Sağa Dön, -1.0 Sola Dön, 0.0
    fire: bool = False         # Tetik çekildi mi?

    def to_movement_tuple(self) -> Tuple[float, float, float]:
        return (self.move_fwd, self.strafe, self.rotate_dir)


class AgentController(ABC):
    """
    Kontrolcü soyut arayüzü (LSP).
    Tüm kontrolcüler (İnsan, Rastgele Bot, PPO Zekası) bu sınıfın yerine geçebilir.
    """
    @abstractmethod
    def get_action(self) -> AgentAction:
        pass


class KeyboardController(AgentController):
    """Klavye tuşlarını AgentAction paketine çevirir."""

    def get_action(self) -> AgentAction:
        keys = pygame.key.get_pressed()
        move_fwd = 0.0
        strafe = 0.0
        rotate_dir = 0.0

        # W - S (İleri - Geri)
        if keys[pygame.K_w]:
            move_fwd += 1.0
        if keys[pygame.K_s]:
            move_fwd -= 1.0

        # D - A (Sağa - Sola Kayma)
        if keys[pygame.K_d]:
            strafe += 1.0
        if keys[pygame.K_a]:
            strafe -= 1.0

        # E/Q veya Sağ/Sol Oklar (Dönüş)
        if keys[pygame.K_e] or keys[pygame.K_RIGHT]:
            rotate_dir += 1.0
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            rotate_dir -= 1.0

        # Space (Ateş)
        fire = keys[pygame.K_SPACE]

        return AgentAction(
            move_fwd=move_fwd,
            strafe=strafe,
            rotate_dir=rotate_dir,
            fire=fire
        )


class RandomController(AgentController):
    """Test amaçlı rastgele hareket üreten sahte kontrolcü."""
    import random

    def get_action(self) -> AgentAction:
        import random
        return AgentAction(
            move_fwd=random.choice([-1.0, 0.0, 1.0]),
            strafe=random.choice([-1.0, 0.0, 1.0]),
            rotate_dir=random.choice([-1.0, 0.0, 1.0]),
            fire=random.random() < 0.05
        )