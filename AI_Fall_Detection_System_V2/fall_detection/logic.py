"""Framework-independent fall-candidate logic and temporal state management."""
from dataclasses import dataclass
from typing import Dict, Optional
import time

@dataclass
class PersonState:
    smoothed_angle: Optional[float] = None
    smoothed_ratio: Optional[float] = None
    candidate_since: Optional[float] = None
    fallen: bool = False
    last_seen: float = 0.0

def is_fall_candidate(torso_angle: float, aspect_ratio: float,
                      angle_threshold: float = 55.0,
                      horizontal_ratio_threshold: float = 1.15) -> bool:
    """Heuristic posture check; not a clinically validated classifier."""
    return torso_angle >= angle_threshold or aspect_ratio >= horizontal_ratio_threshold

class TemporalFallTracker:
    """Smooth per-person measurements and require sustained candidate posture."""
    def __init__(self, confirmation_seconds=0.8, smoothing=0.35,
                 angle_threshold=55.0, horizontal_ratio_threshold=1.15,
                 stale_after_seconds=3.0):
        self.confirmation_seconds = confirmation_seconds
        self.smoothing = smoothing
        self.angle_threshold = angle_threshold
        self.horizontal_ratio_threshold = horizontal_ratio_threshold
        self.stale_after_seconds = stale_after_seconds
        self.people: Dict[int, PersonState] = {}

    def update(self, person_id: int, angle: float, ratio: float, now=None):
        now = time.monotonic() if now is None else now
        state = self.people.setdefault(person_id, PersonState())
        a = self.smoothing
        if state.smoothed_angle is None:
            state.smoothed_angle, state.smoothed_ratio = angle, ratio
        else:
            state.smoothed_angle = a * angle + (1-a) * state.smoothed_angle
            state.smoothed_ratio = a * ratio + (1-a) * state.smoothed_ratio
        state.last_seen = now
        candidate = is_fall_candidate(state.smoothed_angle, state.smoothed_ratio,
                                      self.angle_threshold, self.horizontal_ratio_threshold)
        just_fell = False
        if candidate:
            if state.candidate_since is None:
                state.candidate_since = now
            if not state.fallen and now - state.candidate_since >= self.confirmation_seconds:
                state.fallen, just_fell = True, True
        else:
            state.candidate_since, state.fallen = None, False
        score = min(.99, max(.01, .6*min(state.smoothed_angle/90, 1) +
                             .4*min(state.smoothed_ratio/2, 1)))
        return state.fallen, just_fell, score, state.smoothed_angle, state.smoothed_ratio

    def remove_stale(self, now=None):
        now = time.monotonic() if now is None else now
        for pid in [p for p, s in self.people.items() if now-s.last_seen > self.stale_after_seconds]:
            del self.people[pid]
