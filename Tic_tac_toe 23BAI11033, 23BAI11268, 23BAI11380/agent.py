import numpy as np
import random
import pickle

class QLearningAgent:
    """
    A Q-learning agent for playing Tic-Tac-Toe.
    """
    def __init__(self, player_id, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the agent.
        :param player_id: The ID of the player (1 or 2).
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Exploration rate.
        """
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            return random.choice(available_actions)
        else:
            # Exploitation
            q_values = [self.get_q_value(state, a) for a in available_actions]
            max_q = max(q_values)
            # Find all actions with the maximum Q-value
            best_actions = [a for i, a in enumerate(available_actions) if q_values[i] == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_available_actions, is_done):
        """
        Updates the Q-value for a given state-action pair using the Bellman equation.
        """
        current_q = self.get_q_value(state, action)
        
        if is_done:
            # For a terminal state, the future reward is 0
            max_next_q = 0.0
        else:
            if next_available_actions.size > 0:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions])
            else:
                max_next_q = 0.0
        
        # Bellman equation: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def save_model(self, filename):
        """Saves the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename):
        """Loads the Q-table from a file."""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Model file {filename} not found. Starting with an empty Q-table.")
