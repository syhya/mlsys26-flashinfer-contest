# ✨ PESAgent

PESAgent is an evolutionary agent designed for long-horizon tasks, implementing a **"Planner-Executor-Summary"** paradigm. By mimicking the exploratory workflow of human researchers, it transforms single-step generation into a continuous evolutionary process.

Key features include:
- **Three-Stage Evolution**: Decomposes the optimization loop into Planning (Direction), Execution (Implementation), and Summarization (Reflection).
- **Island Model Evolution**: Supports concurrent evolution across multiple "islands" with migration mechanisms to maintain diversity and escape local optima.
- **Advanced Memory**: Utilizes a combination of MAP-Elites and Boltzmann sampling to manage the population of solutions efficiently.

## Core Components

- **Planner**: Acts as the strategist. It analyzes the global evolutionary state and historical trajectory to propose high-value directions for the next iteration.
- **Executor**: Acts as the engineer. It implements the planner's suggestions, generates code/solutions, runs self-tests, and submits the result for evaluation.
- **Summary**: Acts as the reviewer. It analyzes the execution results, extracts insights (successes/failures), updates the evolutionary memory, and refines the knowledge base.

---

## 🚀 Quick Start

PESAgent includes built-in examples. You can run the `packing_circle_in_unit_square` task to see it in action:

```bash
# Run the example task (results will be in ./output)
./run_math.sh packing_circle_in_unit_square --background

# Stop the task
./run_math.sh stop packing_circle_in_unit_square
```

## 🛠️ Defining Custom Tasks

To define a new evolutionary task, create a folder in `agents/math_agent/examples/<your_task>` with three required files:

1.  **`task_config.yaml`**: Configuration for the task, LLM, and evolutionary parameters.
2.  **`initial_program.py`**: A valid starting solution (can be a dummy implementation).
3.  **`eval_program.py`**: The evaluator logic to score solutions.

### 1. Task Configuration (`task_config.yaml`)

You can configure the evolution process, including concurrency and the island model.

```yaml
# 1. Global LLM Configuration
llm_config:
  model: "deepseek-r1-250528"
  url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  temperature: 0.8
  max_tokens: 32768

# 2. Evolution Process Configuration
evolve:
  task: "Find the optimal configuration for..."  # Your task description
  target_score: 1.0                              # Stop when this score is reached
  max_iterations: 100                            # Maximum number of evolution loops
  concurrency: 5                                 # Number of concurrent workers (parallel evolution)

  # Database & Population Settings (Island Model)
  database:
    storage_type: "in_memory"      # or "redis"
    num_islands: 3                 # Number of parallel populations
    population_size: 100           # Solutions per island
    migration_interval: 10         # Exchange solutions every N iterations
    checkpoint_interval: 50        # Auto-save checkpoint every N iterations

  # Component Selection
  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"

  # Evaluator Settings
  evaluator:
    timeout: 60            # Seconds allowed for evaluation
    evaluate_code: |       # Optional: Inline evaluation logic or path
      from eval_program import evaluate
```

### 2. Initial Program (`initial_program.py`)

Must provide the entry point function expected by the evaluator.

```python
import numpy as np

def solve():
    """Initial valid (but likely suboptimal) solution."""
    return np.array([0, 0, 0])
```

### 3. Evaluation Program (`eval_program.py`)

The heart of evolution. It must return a score (0.0 to 1.0) and feedback.

```python
def evaluate(solution_code):
    # Dynamic import or execution of solution_code
    # ...
    score = calculate_score(result)
    return {
        "score": score,
        "feedback": "The solution is valid but convergence is slow."
    }
```

---

## 💾 Checkpoints & Resuming

PESAgent automatically saves checkpoints based on `checkpoint_interval`.

- **Checkpoints** are stored in the `output/database` directory.
- **Naming format**: `checkpoint-iter-{iteration_id}-{completion_count}`.

To resume from a checkpoint, you typically pass the checkpoint path when initializing `PESAgent` (or via the `run_task.sh` script if supported):

```python
agent = PESAgent(config=config, checkpoint_path="path/to/checkpoint-iter-100-50")
```

---

## 🎩 Advanced Usage: Custom Components

You can customize the **Planner**, **Executor**, or **Summary** by implementing the `Worker` interface and registering them.

```python
from loongflow.framework.evolve import PESAgent

# 1. Initialize Agent
agent = PESAgent(config=config)

# 2. Register Custom Workers
agent.register_planner_worker("my_planner", MyCustomPlanner)
agent.register_executor_worker("my_executor", MyCustomExecutor)

# 3. Run
await agent.run()
```

### Directory Structure

```
├── agents
│   ├── math_agent
│   │   ├── examples
│   │   │   ├── <task_name>
│   │   │   │   ├── eval_program.py
│   │   │   │   ├── initial_program.py
│   │   │   │   └── task_config.yaml
```
