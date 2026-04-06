# Reinforcement Learning Tic-Tac-Toe

A Python implementation of Tic-Tac-Toe using a Reinforcement Learning agent (Q-learning). This project demonstrates how an agent can learn to play the game optimally by playing against itself.

## Project Structure

- **`env.py`**: Contains the `TicTacToeEnv` class, representing the game board, player turns, winning logic, and reward structure.
- **`agent.py`**: Contains the `QLearningAgent` class, which implements the Q-learning algorithm, state-action pair storage (Q-table), and exploration strategy.
- **`train.py`**: A script to train two RL agents against each other over a specified number of games (default 100,000).
- **`play.py`**: A script that allows a human player to play against the trained RL agent.

## How to Use

### 1. Requirements

- Python 3.x
- `numpy` library

You can install `numpy` using pip:

```bash
pip install numpy
```

### 2. Training the Agent

To train the reinforcement learning agent, run:

```bash
python train.py
```

This will play 100,000 games of self-play to populate the Q-table. The resulting models will be saved as `agent1_q_table.pkl` and `agent2_q_table.pkl`.

### 3. Playing Against the Agent

Once the training is complete, you can play against the trained agent by running:

```bash
python play.py
```

You will be playing as **O** (Player 2), and the trained agent will be playing as **X** (Player 1).

## Reinforcement Learning Approach

The agent uses **Q-learning**, a model-free reinforcement learning algorithm. It maintains a **Q-table** where each entry represents a state-action pair and its expected future reward.

- **State**: The 3x3 grid configuration.
- **Action**: The index (0-8) of the empty spot where a player places their symbol.
- **Rewards**:
  - Win: +1
  - Draw: +0.5
  - Loss: -1
  - Continuing move: 0
- **Learning**: The agent updates its Q-values using the Bellman equation after each turn.
- **Exploration**: During training, the agent uses an **epsilon-greedy** policy (with $\epsilon = 0.2$) to explore different strategies. During play, the agent uses a pure exploitation strategy ($\epsilon = 0$).

## License

MIT License
