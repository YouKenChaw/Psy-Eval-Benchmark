from typing import Dict
from .base import EvaluationMethod


class EvaluationManager:
    def __init__(self):
        self.methods: Dict[str, EvaluationMethod] = {}

    def register(self, method: EvaluationMethod):
        self.methods[method.get_name()] = method

    def run(self, dialogue: str, client_profile: str = None) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, method in self.methods.items():
            results[name] = method.evaluate(dialogue, client_profile)
        return results
