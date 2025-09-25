import json
import re
from typing_extensions import override

from openai import OpenAI
from manager.base import EvaluationMethod
from utils import load_prompt


class MusPsy_Rating(EvaluationMethod):
    def __init__(self, args=None):
        self.gpt_api = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name

    def _get_four_dimensions_prompt(self, dialogue: str) -> str:
        """四个维度评估的prompt"""
        prompt_template = load_prompt("MusPsy", "four_dimensions")
        return prompt_template.format(dialogue=dialogue)

    def _get_wai_prompt(self, dialogue: str) -> str:
        """WAI量表评估的prompt"""
        prompt_template = load_prompt("MusPsy", "wai")
        return prompt_template.format(dialogue=dialogue)

    def _get_panas_prompt(self, dialogue: str, profile: str) -> str:
        """PANAS量表评估的prompt"""
        prompt_template = load_prompt("MusPsy", "panas")
        return prompt_template.format(dialogue=dialogue, profile=profile)

    def _parse_four_dimensions_response(self, response: str) -> dict:
        """解析四个维度的响应"""
        scores = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if 'Helpfulness:' in line:
                match = re.search(r'Helpfulness:\s*(\d+)', line)
                if match:
                    scores['helpfulness'] = int(match.group(1))
            elif 'Coherence:' in line:
                match = re.search(r'Coherence:\s*(\d+)', line)
                if match:
                    scores['coherence'] = int(match.group(1))
            elif 'Empathy:' in line:
                match = re.search(r'Empathy:\s*(\d+)', line)
                if match:
                    scores['empathy'] = int(match.group(1))
            elif 'Guidance:' in line:
                match = re.search(r'Guidance:\s*(\d+)', line)
                if match:
                    scores['guidance'] = int(match.group(1))
        
        return scores

    def _parse_wai_response(self, response: str) -> dict:
        """解析WAI量表的响应"""
        scores = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            match = re.search(r'(\d+):\s*(\d+)', line)
            if match:
                question_num = int(match.group(1))
                score = int(match.group(2))
                scores[f'wai_{question_num}'] = score
        
        # 计算WAI总分
        if scores:
            scores['wai_total'] = sum(scores.values())
        
        return scores

    def _parse_panas_response(self, response: str) -> dict:
        """解析PANAS量表的响应"""
        scores = {}
        emotions = ['Interested', 'Excited', 'Strong', 'Enthusiastic', 'Proud', 
                   'Alert', 'Inspired', 'Determined', 'Attentive', 'Active',
                   'Distressed', 'Upset', 'Guilty', 'Scared', 'Hostile', 
                   'Irritable', 'Ashamed', 'Nervous', 'Jittery', 'Afraid']
        
        lines = response.strip().split('\n')
        
        for line in lines:
            for emotion in emotions:
                if emotion.lower() in line.lower():
                    match = re.search(r'(\d+)', line)
                    if match:
                        scores[f'panas_{emotion.lower()}'] = int(match.group(1))
                        break
        
        # 计算正面情绪和负面情绪总分
        positive_emotions = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 
                           'alert', 'inspired', 'determined', 'attentive', 'active']
        negative_emotions = ['distressed', 'upset', 'guilty', 'scared', 'hostile', 
                           'irritable', 'ashamed', 'nervous', 'jittery', 'afraid']
        
        positive_total = sum(scores.get(f'panas_{emotion}', 0) for emotion in positive_emotions)
        negative_total = sum(scores.get(f'panas_{emotion}', 0) for emotion in negative_emotions)
        
        scores['panas_positive_total'] = positive_total
        scores['panas_negative_total'] = negative_total
        scores['panas_total'] = positive_total + negative_total
        
        return scores

    @override
    def evaluate(self, dialogue: str, profile: str = None) -> dict[str, float]:
        """评估对话质量"""
        results = {}
        
        try:
            # 1. 四个维度评估
            four_dim_prompt = self._get_four_dimensions_prompt(dialogue)
            four_dim_response = self.gpt_api.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": four_dim_prompt}],
                temperature=0.0
            ).choices[0].message.content
            
            four_dim_scores = self._parse_four_dimensions_response(four_dim_response)
            results.update(four_dim_scores)
            
            # 2. WAI量表评估
            wai_prompt = self._get_wai_prompt(dialogue)
            wai_response = self.gpt_api.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": wai_prompt}],
                temperature=0.0
            ).choices[0].message.content
            
            wai_scores = self._parse_wai_response(wai_response)
            results.update(wai_scores)
            
            # 3. PANAS量表评估
            if profile:
                panas_prompt = self._get_panas_prompt(dialogue, profile)
                panas_response = self.gpt_api.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": panas_prompt}],
                    temperature=0.0
                ).choices[0].message.content
                
                panas_scores = self._parse_panas_response(panas_response)
                results.update(panas_scores)
        
        except Exception as e:
            print(f"Error in MusPsy evaluation: {e}")
            # 返回默认分数以避免程序崩溃
            return {'error': 0.0}
        
        # 将所有值转换为float类型
        return {k: float(v) for k, v in results.items()}