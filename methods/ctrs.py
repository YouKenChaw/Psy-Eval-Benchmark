from typing import Any, Dict
import re

from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class CTRS_Rating(EvaluationMethod):
    def __init__(self):
        self.gpt_api = OpenAI(
            api_key="sk-EKMrhZekAxRdQRiJ0d5b65Db0700462dB4F8465481Fc03C5",
            base_url="https://api.lmtchina.com/v1"
        )

    def evaluate(self, dialogue: Any, profile: Any = None) -> Dict[str, float]:
        criteria_list = ["understanding", "interpersonal_effectiveness", "collaboration", "guided_discovery", "focus",
                         "strategy"]
        scores = []
        for criteria in criteria_list:
            prompt = load_prompt("ctrs", criteria).format(**{
                "conversation": dialogue
            })
            criteria_output = self.gpt_api.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            ).choices[0].message.content
            try:
                score = int(criteria_output.split(",")[0])
            except Exception as e:
                score = 0
            scores.append(score)

        outputs = dict(zip(criteria_list, scores))
        outputs["sum."] = sum(scores)
        return outputs

    def get_name(self) -> str:
        return "CTRS_Rating"
