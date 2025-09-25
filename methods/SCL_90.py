from typing import Any, Dict
import json
from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class SCL_90_Rating(EvaluationMethod):
    def __init__(self, args):
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name
        self.SCL_90 = load_prompt("SCL_90", "SCL_90")

    def evaluate(self, dialogue: dict, profile: dict = None, use_all_sessions: bool = True) -> Dict[str, Any]:
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
                            # 只取 </strategy></think> 后的内容
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

        # 读取 prompt，并替换 {diag} 占位符
        prompt_template = load_prompt("SCL_90", "SCL_90")
        # 结合 intake_form 分析
        intake_form_str = json.dumps(profile, ensure_ascii=False)
        prompt = prompt_template.replace("<text>", session_text).replace("<intake_form>", intake_form_str)

        # 调用 GPT 接口
        criteria_output = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            # 不设置 temperature
        ).choices[0].message.content

         # 解析 JSON
        try:
            parsed = json.loads(criteria_output)

            diff_scores = {}
            if "difference" in parsed:
                if isinstance(parsed["difference"], list):
                    # 旧格式：list of dict
                    for item in parsed["difference"]:
                        name = item.get("symptom") or item.get("症状名称")
                        score = item.get("score_diff") or item.get("症状差值评分")
                        if name is not None and score is not None:
                            diff_scores[name] = score
                elif isinstance(parsed["difference"], dict):
                    # 新格式：直接是字典
                    diff_scores = parsed["difference"]

        except Exception as e:
            print("JSON parse failed:", e, criteria_output)
            diff_scores = {}

        return {
            "difference": diff_scores
        }




    def get_name(self) -> str:
        return "SCL_90_Rating"
