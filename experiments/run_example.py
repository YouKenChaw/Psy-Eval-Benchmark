import json
import argparse

from manager.evaluation_manager import EvaluationManager
from methods.panas import PANAS_Rating


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    manager = EvaluationManager()
    manager.register(PANAS_Rating(args))

    client_profile = str({
        "info": {
            "age": "未提及",
            "name": "小Q",
            "gender": "男",
            "occupation": "大学二年级学生",
            "educational_background": "大学二年级学生",
            "marital_status": "未提及",
            "family_status": "来自北方城市，非独生子女，还有一个小12岁的弟弟，小Q很爱弟弟。自小父母忙于工作，多与爷爷奶奶生活。",
            "social_status": "在大学交到了两三个好朋友，与朋友沟通交流较多。",
            "medical_history": "无重大躯体疾病史，家庭成员无精神病史。进入大学后的第一个假期曾在医院确诊为抑郁症，并服药治疗。治疗2个月后，再次前往医院诊疗，诊断为抑郁状态。近期自述睡眠质量差，有时会惊醒，醒来浑身无力，白天没有精力学习。",
            "language_features": "表达清晰，说话时语速较慢，逻辑连贯。举止得体，很有礼貌。非常渴望获得帮助，改变现状。"
        },
        "main_problem": "小Q自觉近期情绪状态较为低落，因为学业压力感到焦虑，因为睡眠质量差导致白天精力不够，上课不能集中精神，学习兴趣逐渐下降，同时自觉对于其他事物和活动的兴趣也有所下降。这些情况让小Q对自己的状态感到无力，并影响正常的大学学习生活。",
        "topic": "情绪管理",
        "core_demands": "缓解小Q因为学业压力产生的不愉快情绪，提高小Q对学习的效能感，增强心理弹性；减少因无法满足父母期待带来的自责抑郁和低自我评价，提升小Q的自尊、自信水平；缓解小Q因为童年经历或与父母沟通交流时产生的抑郁情绪，增强小Q在与父母关系中的安全感和掌控感。",
        "Growth Experience": "1. 婴幼期（0～6岁）：基本和爷爷奶奶生活在一起，奶奶爱讲鬼故事，或是说“不要你了”的话，小Q常感到恐惧，让小Q觉得自己是不被奶奶喜欢的。2. 童年期（6～12岁）：父母基本每周回家都会因不满小Q的某些行为而对小Q施加暴力，有时爷爷、奶奶、爸爸、妈妈四人会联合一起责备小Q，小Q感到被孤立，小Q觉得这个家没有人喜欢自己。3. 青春期（13～18岁）：学习压力很大，学习成绩有所下降，经常睡不着觉，很焦虑，感觉痛苦时会用小刀划手臂和大腿。父母认为小Q可以更加优秀，并喜欢拿别人和小Q比较来挖苦嘲讽小Q。高考的成绩不理想，觉得父母对自己非常失望，产生自责愧疚感，这会让小Q觉得自己非常无能。4. 青年期（18～20岁）：学业受挫，情绪持续低落，并伴随自伤行为，期待着能够因意外死去，并签署了遗体捐献协议。",
        "Core_brief": "无能",
        "Special Situation": "1. 高考失利，没有达到父母的要求，让父母失望了（在“无能”的核心信念下，将成绩与自我价值直接绑定，触发自责和内疚）。2. 学业屡次受挫，包括考试不通过、竞赛失利等（将单次挫败解释为“我很无能”，诱发抑郁、焦虑及自伤行为和轻生意念）。3. 面对父母对自己的无限期望或与父母沟通交流时（触发“我必须达成父母对我的期望，我不能让父母失望”，在“无能”信念驱动下产生强烈自我贬低）。4. 被提醒到正向证据时（如一门课程考得不错、摄影方面的才能得到赏识），仍坚持负面评价（在“无能”的核心信念影响下，否定积极信息以维持既有自我图式）。",
        "cognitive pattern": "All-or-Nothing Thinking（“只要没有达到父母的要求就是自己无能”；诱发事件：学业屡次受挫、考试分数没达到期望值）。Overgeneralization（“我学不好这门课，我什么都做不好”“我没有办法胜任大学的学习生活，我很无能”；诱发事件：考试没有考好）。Disqualifying the Positive（“是，但是没有我本应该考得好”“摄影我现在也拍得不好，也提不起兴趣来”；诱发事件：被指出一门课程考得不错或摄影方面的才能得到赏识时）。Catastrophizing（反复出现轻生意念，幻想通过一场意外死去；诱发事件：竞赛失利、考试不通过等挫败后痛苦高涨）。Mental Filtering（只盯着未达期望的成绩与父母失望的表情，忽略同学与哥哥的正向评价；诱发事件：考试后反复自责）。External Validation（“我必须达成父母对我的期望”“我不能让父母失望”；诱发事件：想到或与父母沟通其期待时）。Fortune Telling（“我没有办法胜任大学的学习生活”；诱发事件：考试不如意后对未来学习进行负性预测）。",
    })
    with open("./data/session1_dialogue.json", "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    dialogue = "\n".join([f"{item['role']}: {item['text']}" for item in dialogues[0]["session_dialogue"]])

    print(manager.run(dialogue=dialogue, client_profile=client_profile))
