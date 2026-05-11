"""
Binary RL (GRPO) - Minimal Implementation
简单的Binary RL实现示例，用于学习和理解核心概念
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Trajectory:
    """执行轨迹"""
    task: str
    steps: List[str]  # Agent的每一步
    output: str  # 最终输出
    reward: float = 0.0  # 奖励值


class SimpleAgent:
    """简单的Agent实现"""
    
    def __init__(self, model_name: str = "gpt-3.5"):
        self.model_name = model_name
        self.step_count = 0
    
    def execute(self, task: str) -> Trajectory:
        """
        执行任务
        
        Args:
            task: 任务描述
        
        Returns:
            Trajectory: 执行轨迹
        """
        # 模拟Agent执行过程
        steps = [
            f"Step 1: 理解任务 - {task}",
            f"Step 2: 分析问题",
            f"Step 3: 生成解决方案",
            f"Step 4: 验证结果"
        ]
        
        output = f"完成任务: {task}"
        
        trajectory = Trajectory(
            task=task,
            steps=steps,
            output=output
        )
        
        self.step_count += 1
        return trajectory


class ProcessRewardModel:
    """过程奖励模型 (PRM)"""
    
    def __init__(self):
        self.good_keywords = ["完成", "成功", "正确"]
        self.bad_keywords = ["失败", "错误", "无法"]
    
    def score(self, trajectory: Trajectory) -> float:
        """
        评分轨迹
        
        Args:
            trajectory: 执行轨迹
        
        Returns:
            float: 奖励值 (-1, 0, 1)
                  1: 好
                  0: 中立
                  -1: 坏
        """
        output = trajectory.output.lower()
        
        # 简单的启发式评分
        good_count = sum(1 for kw in self.good_keywords if kw in output)
        bad_count = sum(1 for kw in self.bad_keywords if kw in output)
        
        if good_count > bad_count:
            return 1.0
        elif bad_count > good_count:
            return -1.0
        else:
            return 0.0


class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(self, agent: SimpleAgent, reward_model: ProcessRewardModel):
        self.agent = agent
        self.reward_model = reward_model
        self.rewards_history = []
        self.baseline = 0.0
    
    def compute_advantage(self, reward: float) -> float:
        """
        计算优势函数
        
        GRPO: advantage = reward - baseline
        baseline是历史平均奖励
        
        Args:
            reward: 当前奖励
        
        Returns:
            float: 优势值
        """
        advantage = reward - self.baseline
        return advantage
    
    def update_baseline(self, reward: float):
        """更新baseline（移动平均）"""
        self.rewards_history.append(reward)
        self.baseline = np.mean(self.rewards_history[-10:])  # 最近10个的平均
    
    def ppo_loss(self, advantage: float, old_prob: float = 0.5, epsilon: float = 0.2) -> float:
        """
        PPO损失函数（简化版）
        
        L = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
        
        Args:
            advantage: 优势值
            old_prob: 旧策略的概率
            epsilon: 裁剪范围
        
        Returns:
            float: 损失值
        """
        # 简化的PPO损失
        ratio = 1.0  # 新旧策略的比率（简化）
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
        loss = -min(ratio * advantage, clipped_ratio * advantage)
        return loss
    
    def train_step(self, task: str) -> Dict:
        """
        单个训练步骤
        
        Args:
            task: 任务描述
        
        Returns:
            Dict: 训练信息
        """
        # 1. Agent执行任务
        trajectory = self.agent.execute(task)
        
        # 2. Reward Model评分
        reward = self.reward_model.score(trajectory)
        trajectory.reward = reward
        
        # 3. 计算优势函数
        advantage = self.compute_advantage(reward)
        
        # 4. 计算损失
        loss = self.ppo_loss(advantage)
        
        # 5. 更新baseline
        self.update_baseline(reward)
        
        return {
            "task": task,
            "reward": reward,
            "advantage": advantage,
            "loss": loss,
            "baseline": self.baseline,
            "trajectory": trajectory
        }
    
    def train(self, tasks: List[str], num_epochs: int = 1) -> List[Dict]:
        """
        训练循环
        
        Args:
            tasks: 任务列表
            num_epochs: 训练轮数
        
        Returns:
            List[Dict]: 训练历史
        """
        history = []
        
        for epoch in range(num_epochs):
            for task in tasks:
                result = self.train_step(task)
                result["epoch"] = epoch
                history.append(result)
                
                print(f"Epoch {epoch}, Task: {task}")
                print(f"  Reward: {result['reward']:.2f}")
                print(f"  Advantage: {result['advantage']:.2f}")
                print(f"  Loss: {result['loss']:.4f}")
                print(f"  Baseline: {result['baseline']:.2f}")
                print()
        
        return history


def main():
    """主函数 - 演示Binary RL训练"""
    
    print("=" * 60)
    print("Binary RL (GRPO) - 最小化实现演示")
    print("=" * 60)
    print()
    
    # 1. 初始化组件
    agent = SimpleAgent()
    reward_model = ProcessRewardModel()
    trainer = GRPOTrainer(agent, reward_model)
    
    # 2. 定义任务
    tasks = [
        "完成数据分析",
        "生成报告",
        "优化代码"
    ]
    
    # 3. 训练
    print("开始训练...")
    print()
    history = trainer.train(tasks, num_epochs=2)
    
    # 4. 总结
    print("=" * 60)
    print("训练总结")
    print("=" * 60)
    print(f"总训练步数: {len(history)}")
    print(f"平均奖励: {np.mean([h['reward'] for h in history]):.2f}")
    print(f"最终Baseline: {trainer.baseline:.2f}")
    print()
    
    # 5. 显示改进趋势
    print("奖励趋势:")
    for i, h in enumerate(history):
        print(f"  Step {i+1}: {h['reward']:+.1f}")


if __name__ == "__main__":
    main()
