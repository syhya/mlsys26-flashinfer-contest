# ✨ PESAgent

PESAgent 是一个面向长程任务设计的进化智能体，采用 **"Planner-Executor-Summary"（规划-执行-总结）** 范式。它借鉴人类研究员的探索性工作模式，将单步生成转化为持续的进化过程。

核心特性包括：
- **三阶段进化**：将优化循环分解为规划（方向）、执行（实施）和总结（反思）三个阶段。
- **岛屿模型进化**：支持在多个“岛屿”上进行并发进化，通过迁移机制保持种群多样性并避免陷入局部最优。
- **先进记忆机制**：结合 MAP-Elites 和玻尔兹曼采样（Boltzmann sampling）来高效管理解决方案种群。

## 🧠 核心组件

- **Planner（规划者）**：充当战略家。负责分析全局进化状态和历史轨迹，为下一次迭代提出高价值的改进方向。
- **Executor（执行者）**：充当工程师。负责实施规划者的建议，生成代码/解决方案，运行自测，并提交结果进行评估。
- **Summary（总结者）**：充当审阅者。负责分析执行结果，提取洞察（成功/失败经验），更新进化记忆，并优化知识库。

---

## 🚀 快速开始

PESAgent 内置了示例任务。您可以运行 `packing_circle_in_unit_square` 任务来体验其功能：

```bash
# 运行示例任务（结果将保存在 ./output 目录中）
./run_math.sh packing_circle_in_unit_square --background

# 停止任务
./run_math.sh stop packing_circle_in_unit_square
```

## 🛠️ 定义自定义任务

要定义一个新的进化任务，请在 `agents/math_agent/examples/<your_task>` 目录下创建一个文件夹，并包含以下三个必需文件：

1.  **`task_config.yaml`**：任务、LLM 和进化参数的配置。
2.  **`initial_program.py`**：一个有效的初始解决方案（可以是占位实现）。
3.  **`eval_program.py`**：用于对解决方案进行打分的评估器逻辑。

### 1. 任务配置 (`task_config.yaml`)

您可以配置进化过程，包括并发数和岛屿模型设置。

```yaml
# 1. 全局 LLM 配置
llm_config:
  model: "deepseek-r1-250528"
  url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  temperature: 0.8
  max_tokens: 32768

# 2. 进化流程配置
evolve:
  task: "Find the optimal configuration for..."  # 您的任务描述
  target_score: 1.0                              # 达到此分数时停止
  max_iterations: 100                            # 最大进化循环次数
  concurrency: 5                                 # 并发工作者数量（并行进化）

  # 数据库与种群设置（岛屿模型）
  database:
    storage_type: "in_memory"      # 或 "redis"
    num_islands: 3                 # 并行种群（岛屿）数量
    population_size: 100           # 每个岛屿的解决方案数量
    migration_interval: 10         # 每 N 次迭代交换一次解决方案
    checkpoint_interval: 50        # 每 N 次迭代自动保存检查点

  # 组件选择
  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"

  # 评估器设置
  evaluator:
    timeout: 60            # 允许的评估秒数
    evaluate_code: |       # 可选：内联评估逻辑或路径
      from eval_program import evaluate
```

### 2. 初始程序 (`initial_program.py`)

必须提供评估器所期望的入口函数。

```python
import numpy as np

def solve():
    """初始有效（但可能次优）的解决方案。"""
    return np.array([0, 0, 0])
```

### 3. 评估程序 (`eval_program.py`)

进化的核心。它必须返回一个分数（0.0 到 1.0）和反馈。

```python
def evaluate(solution_code):
    # 动态导入或执行 solution_code
    # ...
    score = calculate_score(result)
    return {
        "score": score,
        "feedback": "解决方案有效，但收敛速度较慢。"
    }
```

---

## 💾 检查点与恢复

PESAgent 会根据 `checkpoint_interval` 自动保存检查点。

- **检查点** 存储在 `output/database` 目录下。
- **命名格式**：`checkpoint-iter-{iteration_id}-{completion_count}`。

要从检查点恢复，通常在初始化 `PESAgent` 时传入检查点路径（或通过 `run_task.sh` 脚本支持）：

```python
agent = PESAgent(config=config, checkpoint_path="path/to/checkpoint-iter-100-50")
```

---

## 🎩 高级用法：自定义组件

您可以通过实现 `Worker` 接口并注册它们来自定义 **Planner**、**Executor** 或 **Summary**。

```python
from loongflow.framework.evolve import PESAgent

# 1. 初始化 Agent
agent = PESAgent(config=config)

# 2. 注册自定义 Worker
agent.register_planner_worker("my_planner", MyCustomPlanner)
agent.register_executor_worker("my_executor", MyCustomExecutor)

# 3. 运行
await agent.run()
```

### 目录结构

```
├── agents
│   ├── math_agent
│   │   ├── examples
│   │   │   ├── <task_name>
│   │   │   │   ├── eval_program.py
│   │   │   │   ├── initial_program.py
│   │   │   │   └── task_config.yaml
```