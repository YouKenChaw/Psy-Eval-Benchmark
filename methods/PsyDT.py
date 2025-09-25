from typing import Any, Dict
import json
from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class PsyDT_Rating(EvaluationMethod):
    def __init__(self, args):
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name
        self.prompt_template = load_prompt("PsyDT", "PsyDT")

    def evaluate(self, dialogue: dict, use_all_sessions: bool = False) -> Dict[str, Any]:
        session_text = ""

        # 拼接对话文本
        sessions = dialogue.get("sessions", [])
        if sessions:
            if use_all_sessions:
                session_texts = []
                for session in sessions:
                    session_lines = []
                    for turn in session.get("session_dialogue", []):
                        text = turn["text"]
                        if turn["role"] == "Counselor":
                            # 去掉 <strategy><think> 标记
                            split_text = text.split("</strategy></think>")
                            text = split_text[-1].strip() if len(split_text) > 1 else text
                        session_lines.append(f'{turn["role"]}: {text}')
                    session_texts.append("\n".join(session_lines))
                session_text = "\n\n".join(session_texts)
            else:
                last_session = sessions[-1]
                session_lines = []
                for turn in last_session.get("session_dialogue", []):
                    text = turn["text"]
                    if turn["role"] == "Counselor":
                        split_text = text.split("</strategy></think>")
                        text = split_text[-1].strip() if len(split_text) > 1 else text
                    session_lines.append(f'{turn["role"]}: {text}')
                session_text = "\n".join(session_lines)

        # 填充 prompt
        prompt = self.prompt_template.replace("{session}", session_text)

        # 调用 GPT
        criteria_output = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        # 解析 JSON
        try:
            parsed = json.loads(criteria_output)
            scores = {dim: val["Score"] for dim, val in parsed.items()}
        except Exception as e:
            print("JSON parse failed:", e, criteria_output)
            scores = {}

        return {
            "scores": scores
        }

    def get_name(self) -> str:
        return "PsyDT_Rating"
