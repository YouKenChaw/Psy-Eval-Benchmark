from abc import ABC, abstractmethod
from typing import Dict, Any


class EvaluationMethod(ABC):
    @abstractmethod
    def evaluate(self, dialogue: Any, profile: Any = None) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
