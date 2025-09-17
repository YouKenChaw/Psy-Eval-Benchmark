# Psy-Eval-Benchmark

## 1.文件目录结构

```
psy-eval-benchmark/        # 项目根目录
│
├── methods/               # 各论文方法实现（每篇论文一个文件/文件夹）
│   ├── __init__.py
│   ├── rule_based_empathy.py   # 示例论文方法
│   ├── bert_classifier.py      # 示例论文方法
│   └── ...
│
├── data/                  # 存放或下载脚本，避免上传大数据
│   ├── README.md          # 描述数据集来源与使用说明
│   └── loaders.py         # 数据加载工具
│
├── manager/               # 管理器 & 接口规范
│   ├── __init__.py
│   ├── base.py            # EvaluationMethod 抽象基类
│   ├── evaluation_manager.py  # 统一运行和对比的管理器
│
├── experiments/           # 实验脚本/配置
│   ├── run_example.py
│   ├── configs/
│   │   └── bert.yaml
│   └── ...
│
├── results/               # 存放输出结果（json/csv/图表）
│   └── ...
│
├── tests/                 # 单元测试
│   ├── test_manager.py
│   ├── test_rule_based.py
│   └── ...
│
├── prompts/                 # 各方法需要的prompt
│   ├── panas/               # 以实现的方法作为文件夹名称
│   │   └── panas_before.txt # 具体的prompt
│
├── requirements.txt       # Python依赖
├── setup.py               # 作为库安装
├── README.md              # 项目介绍（目标、接口示例、使用方法）
└── LICENSE
```

## 2. 代码规范

```
from abc import ABC, abstractmethod
from typing import Any, Dict

class EvaluationMethod(ABC):
    """所有心理咨询评测方法的基类"""

    @abstractmethod
    def evaluate(self, dialogue: Any) -> Dict[str, float]:
        """对输入对话进行评测，返回指标字典"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回方法名称，方便追踪"""
        pass
```

