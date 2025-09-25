from typing import Any, Dict
import json
from openai import OpenAI

from manager.base import EvaluationMethod
from utils import load_prompt


class HTAIS_Rating(EvaluationMethod):
    def __init__(self, args):
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name
        self.HTAIS = load_prompt("HTAIS", "HTAIS")

    def evaluate(self, dialogue_data: dict, use_all_sessions: bool = False) -> Dict[str, Any]:
        """
        对话评估函数，返回紧凑 JSON 分数，同时返回 session_text 和 intake_form。

        Args:
            dialogue (dict): 包含多个 session 的对话数据
            use_all_sessions (bool, optional): 是否分析整个 dialogue，默认 False，只分析最后一个 session

        Returns:
            Dict[str, Any]: 包含 26 个条目的评分字典、session_text 和 intake_form
        """

        dialogue = dialogue_data

        # profile 改为取最后一个 session → session_summary → client_info_get
        sessions = dialogue.get("sessions", [])
        if sessions:
            last_session = sessions[-1]
            profile = last_session.get("session_summary", {}).get("client_info_get", {})
        else:
            profile = {}

        session_text = ""

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
        prompt_template = load_prompt("HTAIS", "HTAIS")
        # 结合 intake_form 分析
        intake_form_str = json.dumps(profile, ensure_ascii=False)
        prompt = prompt_template.replace("{diag}", session_text).replace("{intake_form}", intake_form_str)

        # 调用 GPT 接口
        criteria_output = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        # 解析 JSON
        try:
            parsed = json.loads(criteria_output)
            # 紧凑输出，不换行
            scores = {f"item_{x['item']}": x["score"] for x in parsed}
        except Exception as e:
            print("JSON parse failed:", e, criteria_output)
            scores = {f"item_{i}": 0 for i in range(1, 27)}

        return {
            "htais": scores
        }



    def get_name(self) -> str:
        return "HTAIS_Rating"
