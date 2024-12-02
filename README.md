# Reinforcement Learning

This repository contains code and resources for reinforcement learning projects.

## Project Structure

- [`optimal_control.ipynb`](optimal_control.ipynb) - Jupyter notebook exploring optimal control theory
- [`predictive_control.ipynb`](predictive_control.ipynb) - Jupyter notebook exploring predictive control theory
- [`MDP.ipynb`](MDP.ipynb) - Jupyter notebook exploring markov decision processes
- [`deep_rl.ipynb`](deep_rl.ipynb) - Exploration of neural nets for function approximation
- [`Discrete.py`](Discrete.py) - Familiariser for the basic FrozenLake environment
- [`custom_frozen_lake.py`](custom_frozen_lake.py) - Custom wrapper for FrozenLake environment with configurable rewards
- [`SampleLearning.py`](SampleLearning.py) - Implementation of Temporal Q-Learning algorithm as described in [`deep_rl.ipynb`](deep_rl.ipynb)




## Getting Started

1. Install uv:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
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
