# Phase 1: Binary RL (GRPO) 学习笔记

## 📖 核心概念

### Binary RL是什么？

Binary RL是OpenClaw-RL的第一种培养方法，基于GRPO（Group Relative Policy Optimization）。

**核心流程**：
```
1. Agent执行任务
2. 用户给反馈：好/坏/中立
3. Process Reward Model (PRM)评分
4. GRPO算法计算优势函数
5. PPO风格的损失函数更新模型
6. Agent改进
```

### 关键概念

**Process Reward Model (PRM)**
- 作用：评判Agent的每一步是否正确
- 输入：Agent的执行过程
- 输出：好/坏/中立的评分
- 用处：给出标量奖励信号

**GRPO (Group Relative Policy Optimization)**
- 简单理解：根据相对好坏调整策略
- 优势：比绝对评分更稳定
- 实现：PPO风格的clipped surrogate loss

**标量奖励**
- 一个数字表示好坏程度
- 简单直接
- 信息量相对较少（相比OPD）

### 为什么用Binary RL？

1. **简单** - 只需要好/坏反馈
2. **快速** - 快速收敛
3. **实用** - 用户容易给反馈
4. **成本低** - 不需要复杂的标注

---

## 🔧 实现思路

### 最小化实现（Minimal Version）

```python
# 1. 定义Agent
class SimpleAgent:
    def __init__(self, model):
        self.model = model
    
    def execute(self, task):
        # 执行任务，返回结果
        return self.model.generate(task)

# 2. 定义Reward Model
class ProcessRewardModel:
    def score(self, trajectory):
        # 评判执行过程
        # 返回：1 (好) / 0 (中立) / -1 (坏)
        return score

# 3. 定义训练循环
def train_step(agent, reward_model, task, feedback):
    # 1. Agent执行
    trajectory = agent.execute(task)
    
    # 2. 获取奖励
    reward = reward_model.score(trajectory)
    
    # 3. 计算优势函数（GRPO）
    advantage = compute_advantage(reward)
    
    # 4. 更新模型（PPO）
    loss = ppo_loss(agent.model, trajectory, advantage)
    loss.backward()
    optimizer.step()

# 4. 主循环
for task in tasks:
    feedback = user_feedback(task)  # 用户反馈
    train_step(agent, reward_model, task, feedback)
```

### 关键实现细节

1. **收集轨迹（Trajectory）**
   - 记录Agent的每一步
   - 包括输入、中间状态、输出

2. **计算奖励**
   - 用Reward Model评分
   - 转换为标量值

3. **计算优势函数**
   - GRPO: advantage = reward - baseline
   - baseline是历史平均奖励

4. **更新策略**
   - PPO损失函数
   - Clipped surrogate loss

---

## 📊 学习进度

- [x] 理解Binary RL概念
- [ ] 搭建基础环境
- [ ] 实现SimpleAgent
- [ ] 实现RewardModel
- [ ] 运行训练循环
- [ ] 测试和优化

---

## 🔗 参考资源

- [OpenClaw-RL GitHub](https://github.com/Gen-Verse/OpenClaw-RL)
- [论文第3.1节 - Binary RL](https://arxiv.org/pdf/2603.10165)
- [GRPO论文](https://arxiv.org/abs/2402.06358)
- [PPO论文](https://arxiv.org/abs/1707.06347)

---

**更新时间**: 2026年5月11日 18:10
