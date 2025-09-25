from abc import ABC, abstractmethod
from typing import Dict, Any


class EvaluationMethod(ABC):
    @abstractmethod
    def evaluate(self, input_case: Any) -> Dict[str, float]:
        pass

    def get_name(self) -> str:
        """返回类名称

        调用被子类继承的方法时，返回子类的类名
        """
        return self.__class__.__name__
