import numpy as np
from env import TicTacToeEnv
from agent import QLearningAgent

def train(episodes=100000):
    """
    Trains two agents against each other to play Tic-Tac-Toe.
    :param episodes: Number of games to play for training.
    """
    env = TicTacToeEnv()
    agent1 = QLearningAgent(player_id=1, alpha=0.1, gamma=0.9, epsilon=0.2)
    agent2 = QLearningAgent(player_id=2, alpha=0.1, gamma=0.9, epsilon=0.2)

    print(f"Starting training for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        is_done = False
        
        # Track previous states and actions for both agents to update rewards
        last_state = {1: None, 2: None}
        last_action = {1: None, 2: None}

        while not is_done:
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2
            available_actions = env.get_available_actions()
            
            # Agent chooses an action
            action = agent.choose_action(state, available_actions)
            
            # Store current state and action for learning later
            last_state[current_player] = state
            last_action[current_player] = action
            
            # Environment step
            next_state, reward, is_done, info = env.step(action)
            
            if is_done:
                # Current player either won or it's a draw
                if reward == 1:  # Current player wins
                    agent.learn(state, action, 1, next_state, env.get_available_actions(), True)
                    # The other player lost
                    other_player_id = 3 - current_player
                    other_agent = agent1 if other_player_id == 1 else agent2
                    if last_state[other_player_id] is not None:
                        other_agent.learn(last_state[other_player_id], last_action[other_player_id], -1, next_state, env.get_available_actions(), True)
                elif reward == 0.5:  # Draw
                    agent.learn(state, action, 0.5, next_state, env.get_available_actions(), True)
                    # Other player also gets draw reward
                    other_player_id = 3 - current_player
                    other_agent = agent1 if other_player_id == 1 else agent2
                    if last_state[other_player_id] is not None:
                        other_agent.learn(last_state[other_player_id], last_action[other_player_id], 0.5, next_state, env.get_available_actions(), True)
            else:
                # If game continues, we can update the other player's Q-value for its previous action
                # because the state has changed.
                other_player_id = 3 - current_player
                other_agent = agent1 if other_player_id == 1 else agent2
                if last_state[other_player_id] is not None:
                    # The reward for the other player is 0 for now as the game continues
                    other_agent.learn(last_state[other_player_id], last_action[other_player_id], 0, next_state, env.get_available_actions(), False)

            state = next_state
            
        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")

    # Save the trained models
    agent1.save_model("agent1_q_table.pkl")
    agent2.save_model("agent2_q_table.pkl")
    print("Training finished. Models saved to agent1_q_table.pkl and agent2_q_table.pkl.")

if __name__ == "__main__":
    train()
