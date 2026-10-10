# src/envs/curriculum.py
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TierConfig:
    tier_id: int
    name: str
    description: str
    maps: List[str]


class MapCurriculum:
    """
    Sub-Curriculum Learning Manager for Phase 1 Navigation.
    
    Structure:
      - Tier 0 (Open space & variable distances): faz_1v0.1 to faz_1v0.5
      - Tier 1 (Basic obstacles & dead ends): faz_1v1.1 to faz_1v1.5
      - Tier 2 (Occluded mazes requiring memory): faz_1v2.1 to faz_1v2.6
    
    Modes:
      - 'random' (Domain Randomization): Alternates randomly among maps in the active tier
        on every episode to prevent trajectory/coordinate overfitting.
      - 'sequential': Trains on one map until mastery before advancing.
    """

    DEFAULT_TIERS: List[TierConfig] = [
        TierConfig(
            tier_id=0,
            name="Tier 0: Open Navigation",
            description="Open paths with environmental variations, obstacle-free steering.",
            maps=[
                "configs/maps/faz_1v0.1.yaml",
                "configs/maps/faz_1v0.2.yaml",
                "configs/maps/faz_1v0.3.yaml",
                "configs/maps/faz_1v0.4.yaml",
                "configs/maps/faz_1v0.5.yaml",
            ],
        ),
        TierConfig(
            tier_id=1,
            name="Tier 1: Obstacles & Dead-Ends",
            description="Basic corridors and dead-end traps requiring obstacle avoidance.",
            maps=[
                "configs/maps/faz_1v1.1.yaml",
                "configs/maps/faz_1v1.2.yaml",
                "configs/maps/faz_1v1.3.yaml",
                "configs/maps/faz_1v1.4.yaml",
                "configs/maps/faz_1v1.5.yaml",
            ],
        ),
        TierConfig(
            tier_id=2,
            name="Tier 2: Complex Mazes",
            description="Deep occluded mazes requiring recurrent temporal memory (GRU).",
            maps=[
                "configs/maps/faz_1v2.1.yaml",
                "configs/maps/faz_1v2.2.yaml",
                "configs/maps/faz_1v2.3.yaml",
                "configs/maps/faz_1v2.4.yaml",
                "configs/maps/faz_1v2.5.yaml",
                "configs/maps/faz_1v2.6.yaml",
            ],
        ),
    ]

    def __init__(
        self,
        tiers: Optional[List[TierConfig]] = None,
        sample_mode: str = "random",
        window_size: int = 30,
        promotion_threshold: float = 0.975,
        demotion_threshold: float = 0.20,
        min_episodes_before_promotion: int = 30,
    ):
        self.tiers = tiers or self.DEFAULT_TIERS
        self.sample_mode = sample_mode
        self.window_size = window_size
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.min_episodes_before_promotion = min_episodes_before_promotion

        self.current_tier_idx = 0
        self.current_map_idx = 0
        self.locked = False  # If True, curriculum does not promote or demote
        self.last_promoted_tier: Optional[int] = None  # Records tier id when promotion occurs

        self.recent_successes: deque = deque(maxlen=window_size)
        self.recent_steps: deque = deque(maxlen=window_size)
        self.total_episodes = 0
        self.episodes_in_current_level = 0

        # Currently active map path for this episode
        self._active_map_path = self.current_tier.maps[0]

    def set_tier(self, tier_idx: int, locked: bool = False):
        """Forces the curriculum to a specific tier, optionally locking it."""
        self.current_tier_idx = max(0, min(tier_idx, len(self.tiers) - 1))
        self.locked = locked
        self._active_map_path = self.current_tier.maps[0]
        self.recent_successes.clear()
        self.recent_steps.clear()
        self.episodes_in_current_level = 0

    @property
    def current_tier(self) -> TierConfig:
        return self.tiers[self.current_tier_idx]

    @property
    def current_map_path(self) -> str:
        return self._active_map_path

    @property
    def current_map_name(self) -> str:
        return os.path.basename(self.current_map_path)

    @property
    def success_rate(self) -> float:
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    @property
    def avg_steps(self) -> float:
        if not self.recent_steps:
            return 0.0
        return sum(self.recent_steps) / len(self.recent_steps)

    def get_episode_map(self) -> str:
        """
        Called at the start of each episode. In 'random' mode, selects a random
        map from the current tier to ensure robust generalization across topologies.
        """
        if self.sample_mode == "random":
            self._active_map_path = random.choice(self.current_tier.maps)
        else:
            self._active_map_path = self.current_tier.maps[self.current_map_idx]
        return self._active_map_path

    def record_episode(self, is_success: bool, steps: int) -> Dict[str, any]:
        """
        Records the outcome of an episode and evaluates advancement.
        """
        self.total_episodes += 1
        self.episodes_in_current_level += 1
        self.recent_successes.append(1.0 if is_success else 0.0)
        self.recent_steps.append(steps)

        promoted = False
        demoted = False

        if (
            not self.locked
            and self.episodes_in_current_level >= self.min_episodes_before_promotion
            and len(self.recent_successes) >= self.window_size
        ):
            if self.success_rate >= self.promotion_threshold:
                promoted = self._advance_level()
            elif (
                self.success_rate <= self.demotion_threshold
                and self.current_tier_idx > 0
            ):
                demoted = self._demote_level()

        return {
            "total_episodes": self.total_episodes,
            "tier_id": self.current_tier.tier_id,
            "tier_name": self.current_tier.name,
            "map_name": self.current_map_name,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "promoted": promoted,
            "demoted": demoted,
        }

    def _advance_level(self) -> bool:
        """Promotes to next tier or map."""
        if self.sample_mode == "random":
            # In random mode, the entire tier is mastered! Advance to next tier.
            if self.current_tier_idx + 1 < len(self.tiers):
                self.last_promoted_tier = self.current_tier.tier_id
                self.current_tier_idx += 1
                new_tier = self.current_tier
                print(
                    f"\n[Curriculum Promotion] *** MASTERED {self.tiers[self.current_tier_idx - 1].name} "
                    f"({self.success_rate:.1%}) -> PROMOTED TO {new_tier.name}! ***\n"
                )
            else:
                return False
        else:
            tier = self.current_tier
            if self.current_map_idx + 1 < len(tier.maps):
                self.current_map_idx += 1
                print(
                    f"[Curriculum Promotion] Advanced to map {self.current_map_name} "
                    f"in {tier.name} (Success Rate: {self.success_rate:.1%})"
                )
            elif self.current_tier_idx + 1 < len(self.tiers):
                self.current_tier_idx += 1
                self.current_map_idx = 0
                new_tier = self.current_tier
                print(
                    f"\n[Curriculum Promotion] *** PROMOTED TO {new_tier.name} ***\n"
                )
            else:
                return False

        self.recent_successes.clear()
        self.recent_steps.clear()
        self.episodes_in_current_level = 0
        return True

    def _demote_level(self) -> bool:
        """Demotes to previous tier if struggling."""
        if self.current_tier_idx > 0:
            self.current_tier_idx -= 1
            prev_tier = self.current_tier
            print(
                f"[Curriculum Demotion] Stepped back to {prev_tier.name} "
                f"(Success Rate: {self.success_rate:.1%})"
            )
            self.recent_successes.clear()
            self.recent_steps.clear()
            self.episodes_in_current_level = 0
            return True
        return False
