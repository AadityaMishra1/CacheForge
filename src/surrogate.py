# src/surrogate.py
"""
Minimal surrogate manager stub for when surrogate_enable=false
"""
from pathlib import Path
from typing import Optional

class SurrogateManager:
    """
    Placeholder surrogate manager for cache replacement policy optimization.
    When surrogate_enable=false, this class provides stub methods.
    """
    def __init__(
        self,
        log_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        min_samples: int = 6,
        retrain_interval: int = 3,
    ):
        self.log_path = log_path
        self.model_path = model_path
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self.model = None

    def maybe_retrain(self, force: bool = False) -> None:
        """Stub method for retraining surrogate model."""
        pass

    def predict(self, features: dict) -> Optional[float]:
        """Stub method for prediction."""
        return None
