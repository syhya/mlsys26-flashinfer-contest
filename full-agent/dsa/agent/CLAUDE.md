# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation & Setup

```bash
# Create virtual environment with uv (recommended)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Or with conda
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests for specific module
pytest tests/framework/evolve/
pytest tests/agentsdk/tools/

# Run with verbose output
pytest -v

# Run tests with coverage
pytest --cov=loongflow
```

### Running Examples

#### General Evolve Agent (Mathematical/Algorithmic Problems)

```bash
# Install example-specific dependencies
uv pip install -r ./agents/math_agent/examples/packing_circle_in_unit_square/requirements.txt

# Run task in background
./run_math.sh packing_circle_in_unit_square --background

# Check logs
tail -f ./agents/math_agent/examples/packing_circle_in_unit_square/run.log

# Stop task
./run_math.sh stop packing_circle_in_unit_square
```

#### ML Evolve Agent (Kaggle Competitions)

```bash
# Initialize ML evolve
./run_ml.sh init

# Run task
./run_ml.sh run ml_example --background

# Check logs
tail -f ./agents/ml_agent/examples/ml_example/agent.log

# Stop task
./run_ml.sh stop ml_example
```

### Visualization

```bash
# Launch real-time evolution tracking dashboard
python agents/math_agent/visualizer/visualizer.py --port 8888 --checkpoint-path output-circle-packing/database/checkpoints
```

## High-Level Architecture

LoongFlow is an expert-grade AI agent framework built around the **PES (Plan-Execute-Summary)** paradigm. Unlike traditional "generate-retry" or "mutate-select" approaches, PES enables structured thinking and learning through:

1. **Plan**: Task decomposition, experience retrieval, and blueprint creation
2. **Execute**: Controlled experimentation with intermediate validation
3. **Summary**: Deep reflection, insight extraction, and memory persistence

### Core Components

- **PESAgent**: Main agent class implementing PES with evolutionary memory
- **ReActAgent**: Tool-based reactive agent for dynamic workflows
- **Memory Systems**: Hybrid memory combining structured knowledge storage with evolutionary memory for cross-task generalization
- **Multi-Island + MAP-Elites**: Preserves solution diversity
- **Adaptive Boltzmann Selection**: Balances exploration and exploitation

### Key Directories

- `src/loongflow/` - Core framework implementation
  - `framework/evolve/` - PES implementation and PESAgent
  - `framework/react/` - ReActAgent implementation
  - `agentsdk/` - SDK tools (TodoReadTool, TodoWriteTool, etc.)
- `agents/` - Domain-specific agent implementations
  - `math_agent/` - Mathematical/algorithmic problem solvers
  - `ml_agent/` - Machine learning competition agents
- `tests/` - Unit and integration tests mirroring src structure
- `agents/*/examples/` - Example tasks with task_config.yaml files

### Configuration

Each task requires a `task_config.yaml` file with LLM configuration:

```yaml
llm_config:
  url: "https://xxxxxx/v1"
  api_key: "******"
  model: "openai/gemini-3-pro-preview" # or other OpenAI-compatible models
```

### Development Notes

- Python 3.12+ is required
- Uses `uv` as the recommended package manager (conda also supported)
- Tests use pytest with asyncio support
- Framework is designed for extensibility: register custom planners/executors/summarizers with `PESAgent`
- Memory system supports both file-based and Redis storage
- Real-time visualization available for monitoring evolution progress
