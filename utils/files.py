import os


def load_prompt(method_name, prompt_name):
    prompt_path = os.path.join("./prompts", method_name, f"{prompt_name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()
