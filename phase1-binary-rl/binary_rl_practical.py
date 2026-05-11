"""
Binary RL - 实际应用示例
展示如何处理真实的用户反馈和改进Agent
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class Feedback(Enum):
    """用户反馈类型"""
    GOOD = 1.0      # 好
    NEUTRAL = 0.0   # 中立
    BAD = -1.0      # 坏


@dataclass
class TrainingExample:
    """训练样本"""
    task: str
    agent_response: str
    user_feedback: Feedback
    explanation: str = ""


class CodeReviewAgent:
    """代码审查Agent - 实际应用示例"""
    
    def __init__(self):
        self.version = 1
        self.patterns_learned = []
    
    def review_code(self, code: str) -> str:
        """
        审查代码
        
        Args:
            code: 代码片段
        
        Returns:
            str: 审查意见
        """
        issues = []
        
        # 检查常见问题
        if "TODO" in code:
            issues.append("发现TODO注释，需要完成")
        
        if len(code.split('\n')) > 50:
            issues.append("代码行数过多，建议拆分函数")
        
        if "except:" in code:
            issues.append("发现裸露的except，应该捕获具体异常")
        
        if not code.strip().startswith("def ") and not code.strip().startswith("class "):
            issues.append("缺少函数或类定义")
        
        if issues:
            return "发现问题:\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            return "代码质量良好，无明显问题"
    
    def improve(self, feedback_list: List[TrainingExample]):
        """
        根据反馈改进Agent
        
        Args:
            feedback_list: 反馈列表
        """
        good_count = sum(1 for ex in feedback_list if ex.user_feedback == Feedback.GOOD)
        bad_count = sum(1 for ex in feedback_list if ex.user_feedback == Feedback.BAD)
        
        print(f"改进信息: 好反馈{good_count}个, 坏反馈{bad_count}个")
        
        if good_count > bad_count:
            self.version += 1
            print(f"[OK] Agent升级到v{self.version}")
        else:
            print("[!] 需要更多改进")


class BinaryRLTrainer:
    """Binary RL训练器 - 实际应用版本"""
    
    def __init__(self, agent):
        self.agent = agent
        self.training_data = []
        self.reward_history = []
    
    def collect_feedback(self, task: str, response: str) -> Tuple[Feedback, str]:
        """
        收集用户反馈（模拟）
        
        Args:
            task: 任务
            response: Agent的响应
        
        Returns:
            Tuple[Feedback, str]: 反馈和解释
        """
        # 这里在实际应用中应该是真实的用户反馈
        # 现在我们模拟一些反馈
        
        if "问题" in response and len(response) > 20:
            return Feedback.GOOD, "反馈详细且有建设性"
        elif "无明显问题" in response:
            return Feedback.NEUTRAL, "反馈准确但不够深入"
        else:
            return Feedback.BAD, "反馈不够有用"
    
    def train_iteration(self, tasks: List[str]) -> List[TrainingExample]:
        """
        单次训练迭代
        
        Args:
            tasks: 任务列表
        
        Returns:
            List[TrainingExample]: 训练样本
        """
        examples = []
        
        for task in tasks:
            # 1. Agent执行
            response = self.agent.review_code(task)
            
            # 2. 收集反馈
            feedback, explanation = self.collect_feedback(task, response)
            
            # 3. 记录样本
            example = TrainingExample(
                task=task,
                agent_response=response,
                user_feedback=feedback,
                explanation=explanation
            )
            examples.append(example)
            
            # 4. 记录奖励
            self.reward_history.append(feedback.value)
            
            # 打印信息
            print(f"\n任务: {task[:30]}...")
            print(f"反馈: {feedback.name} ({explanation})")
        
        return examples
    
    def train(self, tasks: List[str], num_iterations: int = 3):
        """
        完整训练流程
        
        Args:
            tasks: 任务列表
            num_iterations: 迭代次数
        """
        print("=" * 70)
        print("Binary RL 训练 - 代码审查Agent")
        print("=" * 70)
        
        for iteration in range(num_iterations):
            print(f"\n【迭代 {iteration + 1}/{num_iterations}】")
            print("-" * 70)
            
            examples = self.train_iteration(tasks)
            self.training_data.extend(examples)
            
            # 改进Agent
            print("\n改进Agent...")
            self.agent.improve(examples)
        
        # 最终总结
        self._print_summary()
    
    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 70)
        print("训练总结")
        print("=" * 70)
        
        total = len(self.reward_history)
        good = sum(1 for r in self.reward_history if r > 0)
        neutral = sum(1 for r in self.reward_history if r == 0)
        bad = sum(1 for r in self.reward_history if r < 0)
        
        print(f"总样本数: {total}")
        print(f"好反馈: {good} ({good/total*100:.1f}%)")
        print(f"中立反馈: {neutral} ({neutral/total*100:.1f}%)")
        print(f"坏反馈: {bad} ({bad/total*100:.1f}%)")
        print(f"平均奖励: {np.mean(self.reward_history):.2f}")
        print(f"Agent最终版本: v{self.agent.version}")


def main():
    """主函数"""
    
    # 1. 初始化Agent和训练器
    agent = CodeReviewAgent()
    trainer = BinaryRLTrainer(agent)
    
    # 2. 定义代码样本（作为任务）
    code_samples = [
        """def process_data(data):
    # TODO: 添加错误处理
    result = []
    for item in data:
        result.append(item * 2)
    return result""",
        
        """def calculate(x, y):
    try:
        return x / y
    except:
        return 0""",
        
        """def hello():
    print("Hello, World!")
    return True"""
    ]
    
    # 3. 训练
    trainer.train(code_samples, num_iterations=2)
    
    # 4. 测试改进后的Agent
    print("\n" + "=" * 70)
    print("测试改进后的Agent")
    print("=" * 70)
    
    test_code = """def new_function(x):
    if x > 0:
        return x ** 2
    else:
        return 0"""
    
    print(f"\n测试代码:\n{test_code}\n")
    print("Agent审查意见:")
    print(agent.review_code(test_code))


if __name__ == "__main__":
    main()
