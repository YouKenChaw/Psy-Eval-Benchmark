from abc import ABC, abstractmethod
from typing import Dict, Any


class EvaluationMethod(ABC):
    @abstractmethod
    def evaluate(self, input_case: Any) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
