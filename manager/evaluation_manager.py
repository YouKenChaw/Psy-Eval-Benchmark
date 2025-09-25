from typing import Dict, Any
from .base import EvaluationMethod


class EvaluationManager:
    def __init__(self):
        self.methods: Dict[str, EvaluationMethod] = {}

    def register(self, method: EvaluationMethod):
        self.methods[method.get_name()] = method

    def run(self, case: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, method in self.methods.items():
            results[name] = method.evaluate(case)
        return results
