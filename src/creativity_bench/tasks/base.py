from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskResult:
    name: str
    score: float  # normalized to [0, 1]
    metrics: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "metrics": self.metrics,
            "details": self.details,
        }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
