# Reinforcement Learning

This repository contains code and resources for reinforcement learning projects.

## Project Structure

- [`custom_frozen_lake.py`](custom_frozen_lake.py) - Custom wrapper for FrozenLake environment with configurable rewards
- [`SampleLearning.py`](SampleLearning.py) - Implementation of Temporal Q-Learning algorithm
- [`Discrete.py`](Discrete.py) - Example of using the basic FrozenLake environment
- [`optimal_control.ipynb`](optimal_control.ipynb) - Jupyter notebook exploring optimal control theory


## Getting Started

1. Install uv:

```sh
pip install uv
```

2. Set up Python environment (requires Python 3.12+):
```sh
uv run
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Usage

Run the random FrozenLake agent to familarise yourself with the env:
```sh
python Discrete.py
```

Run the temporal Q-learning example:

```sh
python SampleLearning.py
```

## Development
This project uses Ruff for code formatting. A GitHub Action will automatically format code on push.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.