from typing import Any, Dict
import json
from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class RRO_Rating(EvaluationMethod):
    def __init__(self, args):
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name
        self.RRO = load_prompt("RRO", "RRO")

    def evaluate(self, dialogue: dict, profile: dict = None, use_all_sessions: bool = False) -> Dict[str, Any]:
        """
        对话评估函数，只返回24个条目的分数。
        """
        session_text = ""

        sessions = dialogue.get("sessions", [])
        if sessions:
            if use_all_sessions:
                # 拼接所有 session 对话，每个 session 用空行分隔
                session_texts = []
                for session in sessions:
                    session_lines = []
                    for turn in session.get("session_dialogue", []):
                        text = turn["text"]
                        if turn["role"] == "Counselor":
                            split_text = text.split("</strategy></think>")
                            text = split_text[-1].strip() if len(split_text) > 1 else text
                        session_lines.append(f'{turn["role"]}: {text}')
                    session_texts.append("\n".join(session_lines))
                session_text = "\n\n".join(session_texts)
            else:
                # 只取最后一个 session
                last_session = sessions[-1]
                session_lines = []
                for turn in last_session.get("session_dialogue", []):
                    text = turn["text"]
                    if turn["role"] == "Counselor":
                        split_text = text.split("</strategy></think>")
                        text = split_text[-1].strip() if len(split_text) > 1 else text
                    session_lines.append(f'{turn["role"]}: {text}')
                session_text = "\n".join(session_lines)

        # 读取 prompt，并结合来访者信息
        prompt_template = load_prompt("RRO", "RRO")
        intake_form_str = json.dumps(profile, ensure_ascii=False)
        prompt = prompt_template.replace("{diag}", session_text).replace("{intake_form}", intake_form_str)

        # 调用 GPT 接口
        criteria_output = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        # 解析 JSON，只保留分数
        try:
            parsed = json.loads(criteria_output)
            items_scores = [x["score"] for x in parsed.get("items", parsed)]
            factor_scores = parsed.get("factor_scores", {
                "Client Realism": 0,
                "Client Genuineness": 0,
                "Therapist Realism": 0,
                "Therapist Genuineness": 0
            })
        except Exception as e:
            print("JSON parse failed:", e, criteria_output)
            items_scores = [0] * 24
            factor_scores = {
                "Client Realism": 0,
                "Client Genuineness": 0,
                "Therapist Realism": 0,
                "Therapist Genuineness": 0
            }

        # 返回结构
        return {
            "RRO": {
                "items": items_scores,
                "factor_scores": factor_scores
            }
        }



    def get_name(self) -> str:
        return "RRO_Rating"
