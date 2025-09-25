import json
import re
from typing_extensions import override
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from openai import OpenAI
from manager.base import EvaluationMethod
from utils import load_prompt


class SoulChat_Rating(EvaluationMethod):
    def __init__(self, args=None):
        self.gpt_api = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
        self.model_name = args.model_name
        
        # 初始化NLTK资源（如果可用）
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            self.use_nltk = True
        except ImportError:
            print("NLTK not available, using simple tokenization")
            self.use_nltk = False


    def _simple_tokenize(self, text: str) -> list[str]:
        """简单的分词方法"""
        if self.use_nltk:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        # 简单的分词方法
        import string
        # 移除标点符号并按空格分割
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower().split()

    def _calculate_bleu_scores(self, reference: str, candidate: str) -> dict[str, float]:
        """计算BLEU-1/2/3/4分数"""
        ref_tokens = self._simple_tokenize(reference)
        cand_tokens = self._simple_tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        # 使用NLTK的BLEU计算（如果可用）
        if self.use_nltk:
            try:
                smoothing = SmoothingFunction().method1
                bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
            except:
                pass
        
        # 简单的BLEU实现
        def calculate_ngram_precision(ref_tokens, cand_tokens, n):
            if len(cand_tokens) < n:
                return 0.0
            
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
            
            matches = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) for ngram in cand_ngrams)
            total = sum(cand_ngrams.values())
            
            return matches / total if total > 0 else 0.0
        
        bleu_1 = calculate_ngram_precision(ref_tokens, cand_tokens, 1)
        bleu_2 = calculate_ngram_precision(ref_tokens, cand_tokens, 2)
        bleu_3 = calculate_ngram_precision(ref_tokens, cand_tokens, 3)
        bleu_4 = calculate_ngram_precision(ref_tokens, cand_tokens, 4)
        
        return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}

    def _calculate_rouge_scores(self, reference: str, candidate: str) -> dict[str, float]:
        """计算ROUGE-1/2/L分数"""
        ref_tokens = self._simple_tokenize(reference)
        cand_tokens = self._simple_tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        
        # ROUGE-1 (基于单词)
        ref_unigrams = Counter(ref_tokens)
        cand_unigrams = Counter(cand_tokens)
        
        overlap_1 = sum(min(ref_unigrams[token], cand_unigrams[token]) for token in cand_unigrams)
        rouge_1_precision = overlap_1 / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        rouge_1_recall = overlap_1 / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        rouge_1 = (2 * rouge_1_precision * rouge_1_recall) / (rouge_1_precision + rouge_1_recall) if (rouge_1_precision + rouge_1_recall) > 0 else 0.0
        
        # ROUGE-2 (基于二元组)
        ref_bigrams = Counter([tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens)-1)])
        cand_bigrams = Counter([tuple(cand_tokens[i:i+2]) for i in range(len(cand_tokens)-1)])
        
        overlap_2 = sum(min(ref_bigrams[bigram], cand_bigrams[bigram]) for bigram in cand_bigrams)
        rouge_2_precision = overlap_2 / len(cand_bigrams) if len(cand_bigrams) > 0 else 0.0
        rouge_2_recall = overlap_2 / len(ref_bigrams) if len(ref_bigrams) > 0 else 0.0
        rouge_2 = (2 * rouge_2_precision * rouge_2_recall) / (rouge_2_precision + rouge_2_recall) if (rouge_2_precision + rouge_2_recall) > 0 else 0.0
        
        # ROUGE-L (最长公共子序列)
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(ref_tokens, cand_tokens)
        rouge_l_precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        rouge_l_recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        rouge_l = (2 * rouge_l_precision * rouge_l_recall) / (rouge_l_precision + rouge_l_recall) if (rouge_l_precision + rouge_l_recall) > 0 else 0.0
        
        return {"rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_l": rouge_l}

    def _extract_last_counselor_response(self, dialogue: str) -> str:
        """提取对话中咨询师的最后一句话"""
        lines = dialogue.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('Counselor:') or line.startswith('counselor:') or line.startswith('咨询师:') or line.startswith('therapist:'):
                return line.split(':', 1)[1].strip()
        # 如果没有找到明确的咨询师标记，返回最后一行
        return lines[-1].strip() if lines else ""

    def _get_reference_response(self, dialogue: str, profile: str = None) -> str:
        """获取参考响应 - 这里使用一个理想的咨询师回应作为参考"""
        # 在实际应用中，这可能是：
        # 1. 专家标注的理想回应
        # 2. 多个专家回应的平均
        # 3. 基于客户档案生成的标准回应
        # 这里我们使用一个通用的高质量咨询师回应作为参考
        return "我理解你的感受，这确实是一个困难的情况。让我们一起探讨一些可能的应对方式，看看什么对你最有帮助。你觉得现在最需要的支持是什么？"

    def _create_cehs_prompt(self, dialogue: str, ai_response: str) -> str:
        """创建CEHS框架的评估prompt"""
        prompt_template = load_prompt("soulchat", "cehs_evaluation")
        return prompt_template.format(
            conversation=dialogue,
            ai_response=ai_response
        )

    @override
    def evaluate(self, dialogue: str, profile: str = None) -> dict[str, float]:
        # 提取咨询师的最后一句话作为AI Response
        ai_response = self._extract_last_counselor_response(dialogue)
        
        # 获取参考响应
        reference_response = self._get_reference_response(dialogue, profile)
        
        # 计算BLEU和ROUGE分数
        bleu_scores = self._calculate_bleu_scores(reference_response, ai_response)
        rouge_scores = self._calculate_rouge_scores(reference_response, ai_response)
        
        # 构建CEHS评估prompt
        prompt = self._create_cehs_prompt(dialogue, ai_response)
        
        try:
            # 调用GPT API进行评估
            response = self.gpt_api.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON结果
            try:
                # 找到JSON部分
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                    result = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                # 如果解析失败，尝试从文本中提取分数
                print(f"JSON parsing failed: {e}")
                print(f"Response text: {result_text}")
                
                # 使用正则表达式提取分数
                content_score = self._extract_score_from_text(result_text, "content_score", 0, 2)
                empathy_score = self._extract_score_from_text(result_text, "empathy_score", 0, 2)
                helpfulness_score = self._extract_score_from_text(result_text, "helpfulness_score", 0, 2)
                safety_score = self._extract_score_from_text(result_text, "safety_score", 0, 1)
                
                result = {
                    "content_score": content_score,
                    "empathy_score": empathy_score,
                    "helpfulness_score": helpfulness_score,
                    "safety_score": safety_score
                }
            
            # 确保所有分数都是数字类型
            scores = {
                "content_score": float(result.get("content_score", 0)),
                "empathy_score": float(result.get("empathy_score", 0)),
                "helpfulness_score": float(result.get("helpfulness_score", 0)),
                "safety_score": float(result.get("safety_score", 0))
            }
            
            # 添加BLEU和ROUGE分数
            scores.update(bleu_scores)
            scores.update(rouge_scores)
            
            # 计算总分（仅基于CEHS分数）
            scores["cehs_total"] = scores["content_score"] + scores["empathy_score"] + scores["helpfulness_score"] + scores["safety_score"]
            
            return scores
            
        except Exception as e:
            print(f"Error in SoulChat evaluation: {e}")
            # 返回默认的0分，但仍包含BLEU和ROUGE分数
            default_scores = {
                "content_score": 0.0,
                "empathy_score": 0.0,
                "helpfulness_score": 0.0,
                "safety_score": 0.0,
                "cehs_total": 0.0
            }
            default_scores.update(bleu_scores)
            default_scores.update(rouge_scores)
            return default_scores

    def _extract_score_from_text(self, text: str, score_name: str, min_val: int, max_val: int) -> float:
        """从文本中提取特定的分数"""
        pattern = rf'"{score_name}":\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            return max(min_val, min(max_val, score))  # 确保分数在有效范围内
        return 0.0