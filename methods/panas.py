from typing import Any, Dict
import re

from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class PANAS_Rating(EvaluationMethod):
    def __init__(self, args):
        self.gpt_api = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name

        self.panas_before = load_prompt("panas", "panas_before")
        self.panas_after = load_prompt("panas", "panas_after")

    def evaluate(self, dialogue: Any, profile: Any = None) -> Dict[str, float]:
        panas_before_inputs = self.panas_before.format(**{
            "intake_form": profile
        })
        panas_before_output = self.gpt_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": panas_before_inputs}],
            temperature=0.0
        ).choices[0].message.content

        panas_after_inputs = self.panas_after.format(**{
            "intake_form": profile,
            "dialogue": dialogue
        })
        panas_after_output = self.gpt_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": panas_after_inputs}],
            temperature=0.0
        ).choices[0].message.content

        pos_be_score, neg_be_score = self.calculate_panas_score(panas_before_output)

        pos_aft_score, neg_aft_score = self.calculate_panas_score(panas_after_output)
        pos_change, neg_change = pos_aft_score - pos_be_score, neg_aft_score - neg_be_score

        return {"positive_emotion_change": round(pos_change, 3), "negative_emotion_change": round(neg_change, 3)}

    def calculate_panas_score(self, output):
        criteria_list = ["Interested", "Excited", "Strong", "Enthusiastic", "Proud", "Alert", "Inspired", "Determined",
                         "Attentive", "Active", "Distressed", "Upset", "Guilty", "Scared", "Hostile", "Irritable",
                         "Ashamed", "Nervous", "Jittery", "Afraid"]
        score_dict = {}
        for cri in criteria_list:
            score_dict[cri] = []

        before_lines = output.split("\n")
        for line in before_lines:
            if not line:
                continue
            criteria = line.split(",")[0].strip()
            try:
                score = int(re.findall(r'\d+', line.split(",")[-1].strip())[0])
            except:
                continue
            if criteria in criteria_list:
                score_dict[criteria].append(score)

        avg_score_dict = {}
        for key in score_dict.keys():
            avg_score_dict[key] = sum(score_dict[key]) / len(score_dict[key])

        positive_score = []
        for key in criteria_list[:10]:
            positive_score.append(avg_score_dict[key])

        negative_score = []
        for key in criteria_list[10:]:
            negative_score.append(avg_score_dict[key])

        positive_criteria_score = sum(positive_score) / len(positive_score)
        negative_criteria_score = sum(negative_score) / len(negative_score)

        return positive_criteria_score, negative_criteria_score

    def get_name(self) -> str:
        return "PANAS_Rating"
