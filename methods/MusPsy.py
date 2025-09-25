import json
import re
from typing_extensions import override

from openai import OpenAI
from manager.base import EvaluationMethod


class MusPsy_Rating(EvaluationMethod):
    def __init__(self, args=None):
        self.gpt_api = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name

    def _get_four_dimensions_prompt(self, dialogue: str) -> str:
        """四个维度评估的prompt"""
        return f"""Please evaluate the quality of the conversation between the counselor and the user based on the given criteria.
Criteria
Helpfulness: Assesses the applicability of explanations and suggestions from a psychological counseling perspective. Rate from 1 to 5, where 1 means not helpful at all and 5 means highly helpful.
Coherence: Evaluates the logical flow and structure of the conversation. Rate from 1 to 5, where 1 means completely incoherent and 5 means well-structured and logically sound.
Empathy: Measures the counselor's ability to understand and respond to the user's feelings. Rate from 1 to 5, where 1 means no empathy at all and 5 means the same level of empathy as a human counselor.
Guidance: Evaluates the specificity and practicality of the counselor's suggestions. Rate from 1 to 5, where 1 means the suggestions are not practical at all and 5 means the suggestions are very specific and feasible.
Format:
Just provide the scores without explanations, and the scores should be integers.
Helpfulness: [score]
Coherence: [score]
Empathy: [score]
Guidance: [score]
[Conversation Record]
{dialogue}"""

    def _get_wai_prompt(self, dialogue: str) -> str:
        """WAI量表评估的prompt"""
        return f"""The following counseling session reflects a dialogue between the client and the counselor. Please evaluate how the client might respond to the questions below after the session by selecting the appropriate scores from the provided rating scale.
Note: Please provide only the question numbers and their respective scores in the specified format. Do not repeat the questions themselves or add unnecessary prefixes or control characters.
[Question List]
1. As a result of these sessions I am clearer as to how I might be able to change.
2. What I am doing in therapy gives me new ways of looking at my problem.
3. I believe counselor likes me.
4. counselor and I collaborate on setting goals for my therapy.
5. counselor and I respect each other.
6. counselor and I are working towards mutually agreed upon goals.
7. I feel that counselor appreciates me.
8. counselor and I agree on what is important for me to work on.
9. I feel counselor cares about me even when I do things that he/she does not approve of.
10. I feel that the things I do in therapy will help me to accomplish the changes that I want.
11. counselor and I have established a good understanding of the kind of changes that would be good for me.
12. I believe the way we are working with my problem is correct.
[Rating Scale]
1: Seldom
2: Sometimes
3: Fairly Often
4: Very Often
5: Always
[Response Format]
Question number: Score
[Below is the history of the counseling dialogue] 
{dialogue}"""

    def _get_panas_prompt(self, dialogue: str, profile: str) -> str:
        """PANAS量表评估的prompt"""
        return f"""A person with the characteristics listed in the intake form received counseling. The following counseling session is a conversation between the client and the counselor. After reviewing the conversation, evaluate the intensity of each of the following feelings the person might have experienced once the counseling session is complete: Interested, Excited, Strong, Enthusiastic, Proud, Alert, Inspired, Determined, Attentive, Active, Distressed, Upset, Guilty, Scared, Hostile, Irritable, Ashamed, Nervous, Jittery, Afraid.
For each feeling, generate a score from 1 to 5 using the following scale:
1 - Very slightly or not at all
2 - A little
3 - Moderately
4 - Quite a bit
5 - Extremely
Additionally, please provide a brief explanation for each score. Output in the specified format without including any irrelevant control characters or prefixes.
Here is the text:
{profile}
Here is the counseling session:
{dialogue}
[Output Format]
Emotion: Score. Explanation"""

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