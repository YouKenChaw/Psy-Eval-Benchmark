import requests
from typing import Dict, Any
from openai import OpenAI

from manager.base import EvaluationMethod
from utils.files import load_prompt


class SAGE(EvaluationMethod):
    def __init__(self, args):
        self.base_url = args.api_base
        self.model = args.model_name
        self.api_key = args.api_key  # 在实际使用中替换为真实的API密钥
        self.prompt_template = load_prompt("sage", "emotion_analysis")

    def extract_prompt_variables(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """从输入数据中提取prompt模板所需的变量"""
        profile = input_data["client_info"]
        session = input_data["sessions"][0]  # ! 需要确认输入是不是选择的第一个session片段

        target = f"{profile.get('main_problem', '')}. {profile.get('core_demands', '')}"

        static_traits = profile["static_traits"]
        simulator_role = f"""
            姓名: {static_traits['name']}
            年龄: {static_traits['age']}
            性别: {static_traits['gender']}
            职业: {static_traits['occupation']}
            教育背景: {static_traits['educational_background']}
            婚姻状况: {static_traits['marital_status']}
            家庭状况: {static_traits['family_status']}
            社会地位: {static_traits['social_status']}
            医疗史: {static_traits['medical_history']}
            语言特征: {static_traits['language_features']}

            主要问题: {profile['main_problem']}
            核心需求: {profile['core_demands']}
            成长经历: {', '.join(profile['growth_experience'])}
            核心简介: {', '.join(profile['core_brief'])}
            特殊情况: {', '.join(profile['special_situation'])}
            """

        summary = session["session_summary"]
        simulator_scene = f"""
            会话摘要: {summary['session_summary_abstract']}
            目标评估: {summary['goal_assessment']['evidence_and_analysis']}
            来访者状态分析:
            - 情感状态: {summary['client_state_analysis']['affective_state']}
            - 认知模式: {summary['client_state_analysis']['cognitive_patterns']}
            - 行为模式: {summary['client_state_analysis']['behavioral_patterns']}
            - 治疗联盟: {summary['client_state_analysis']['therapeutic_alliance']}
            - 未解决点: {summary['client_state_analysis']['unresolved_points_or_tensions']}
            """

        affective_state = summary['client_state_analysis']['affective_state']
        emotion = self.map_affective_state_to_emotion(affective_state)

        dialog_history = ""
        for dialog in session["session_dialogue"]:
            role = "咨询师" if dialog["role"] == "Counselor" else "来访者"
            dialog_history += f"{role}: {dialog['text']}\n"

        return {
            "target": target,
            "simulator_role": simulator_role,
            "simulator_scene": simulator_scene,
            "emotion": str(emotion),
            "dialog_history": dialog_history
        }

    def map_affective_state_to_emotion(self, affective_state: str) -> int:
        """将情感状态描述映射到0-100的emotion数值"""
        affective_state_lower = affective_state.lower()

        if any(word in affective_state_lower for word in ['积极', '乐观', '愉快', '兴奋', '满意']):
            return 80
        elif any(word in affective_state_lower for word in ['平静', '稳定', '中性', '正常']):
            return 60
        elif any(word in affective_state_lower for word in ['低落', '消极', '焦虑', '抑郁', '紧张']):
            return 40
        elif any(word in affective_state_lower for word in ['严重', '极度', '崩溃', '绝望']):
            return 20
        else:
            return 50

    def build_prompt(self, variables: Dict[str, str]) -> str:
        """构建完整的prompt"""
        return self.prompt_template.format(**variables)

    def call_llm_api(self, prompt: str) -> str:
        """调用LLM API"""
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.7
        )

        return response.choices[0].message.content

    def parse_evaluation_result(self, result: str) -> Dict[str, str]:
        """解析评估结果"""
        parsed_result = {}
        lines = result.split('\n')

        current_key = None
        current_value = []

        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                if current_key:
                    parsed_result[current_key] = '\n'.join(current_value).strip()
                current_key = line[:-1]
                current_value = []
            elif current_key and line:
                current_value.append(line)

        if current_key and current_value:
            parsed_result[current_key] = '\n'.join(current_value).strip()

        return parsed_result

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """实现基类的evaluate方法"""
        variables = self.extract_prompt_variables(input_data)
        prompt = self.build_prompt(variables)
        evaluation_result = self.call_llm_api(prompt)
        parsed_result = self.parse_evaluation_result(evaluation_result)
        # ToDo：返回内容确认！
        return {
            "prompt_variables": variables,
            "generated_prompt": prompt,
            "raw_evaluation_result": evaluation_result,
            "parsed_evaluation": parsed_result
        }

    def get_name(self) -> str:
        return "sage"
