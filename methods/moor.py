from typing import Any, Dict
import json
from openai import OpenAI
import os

from manager.base import EvaluationMethod
from utils import load_prompt


class moor_Rating(EvaluationMethod):
    def __init__(self, args):
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name

        # 定义多个 prompt 文件路径（位于 moor 文件夹中）
        self.prompt_files = [
            "Interaction.txt",
            "Perception.txt",
            "Intervention.txt",
            "Ethics.txt"
        ]

        # 预加载所有 prompt 文件
        self.prompts = {}
        for file in self.prompt_files:
            prompt_name = os.path.splitext(file)[0]  # 去掉扩展名
            self.prompts[prompt_name] = load_prompt("moor", prompt_name)

    def _build_session_text(self, dialogue: dict, use_all_sessions: bool = True) -> str:
        """拼接对话文本"""
        session_text = ""
        sessions = dialogue.get("sessions", [])

        if not sessions:
            return session_text

        def clean_text(turn_text: str, role: str) -> str:
            if role == "Counselor":
                split_text = turn_text.split("</strategy></think>")
                return split_text[-1].strip() if len(split_text) > 1 else turn_text
            return turn_text.strip()

        if use_all_sessions:
            session_blocks = []
            for session in sessions:
                lines = [
                    f'{turn["role"]}: {clean_text(turn["text"], turn["role"])}'
                    for turn in session.get("session_dialogue", [])
                ]
                session_blocks.append("\n".join(lines))
            session_text = "\n\n".join(session_blocks)
        else:
            last_session = sessions[-1]
            lines = [
                f'{turn["role"]}: {clean_text(turn["text"], turn["role"])}'
                for turn in last_session.get("session_dialogue", [])
            ]
            session_text = "\n".join(lines)

        return session_text

    def evaluate(self, dialogue: dict, use_all_sessions: bool = True) -> Dict[str, Any]:
        session_text = self._build_session_text(dialogue, use_all_sessions)
        all_scores = {}
        average_scores = {}

        # 针对每个维度 prompt 调用 GPT
        for prompt_name, prompt_template in self.prompts.items():
            prompt = prompt_template.replace("{session}", session_text)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content

            try:
                parsed = json.loads(response)

                # ✅ 针对模型返回 list 的情况解析
                if isinstance(parsed, list) and all("score" in x for x in parsed):
                    scores = [x["score"] for x in parsed]
                    # 记录到 all_scores
                    for x in parsed:
                        all_scores[f"{prompt_name}_item_{x['item']}"] = x["score"]

                    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
                    average_scores[prompt_name] = avg_score

                # 兼容旧的 dict 格式
                elif isinstance(parsed, dict):
                    scores = []
                    for k, v in parsed.items():
                        if isinstance(v, dict) and "Score" in v:
                            scores.append(v["Score"])
                            all_scores[f"{prompt_name}_{k}"] = v["Score"]
                        elif isinstance(v, (int, float)):
                            scores.append(v)
                            all_scores[f"{prompt_name}_{k}"] = v
                    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
                    average_scores[prompt_name] = avg_score

                else:
                    print(f"[{prompt_name}] Unrecognized format:", response)
                    average_scores[prompt_name] = 0

            except Exception as e:
                print(f"[{prompt_name}] JSON parse failed:", e, response)
                average_scores[prompt_name] = 0

        return {
            "scores": all_scores,
            "average_scores": average_scores
        }

    def get_name(self) -> str:
        return "moor_Rating"


