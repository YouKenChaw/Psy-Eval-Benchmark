import json
import argparse

from manager.evaluation_manager import EvaluationManager
from methods.panas import PANAS_Rating
from methods.sage import SAGE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()

    manager = EvaluationManager()
    manager.register(SAGE(args))

    with open("./data/case.json", "r", encoding="utf-8") as f:
        input_case = json.load(f)

    print(manager.run(case=input_case))
