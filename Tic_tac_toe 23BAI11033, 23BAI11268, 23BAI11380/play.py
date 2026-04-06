import numpy as np
from env import TicTacToeEnv
from agent import QLearningAgent

def play_game():
    """
    Main function to play Tic-Tac-Toe against the trained agent.
    """
    env = TicTacToeEnv()
    # Initialize the agent as Player 1 (X) with no exploration (epsilon=0)
    agent = QLearningAgent(player_id=1, epsilon=0)
    
    # Try to load the trained model
    try:
        agent.load_model("agent1_q_table.pkl")
        print("Trained model loaded successfully.")
    except FileNotFoundError:
        print("No trained model found. Please run train.py first to train the agent.")
        return

    while True:
        state = env.reset()
        is_done = False
        print("\n" + "="*20)
        print("TIC-TAC-TOE: YOU vs AGENT")
        print("="*20)
        print("You are O (Player 2), Agent is X (Player 1).")
        print("The board positions are as follows:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 ")
        
        env.render()

        while not is_done:
            current_player = env.current_player
            if current_player == 1:
                # Agent's turn
                print("Agent (X) is thinking...")
                available_actions = env.get_available_actions()
                action = agent.choose_action(state, available_actions)
                state, reward, is_done, info = env.step(action)
                print(f"Agent chose position {action}")
            else:
                # Human's turn
                available_actions = env.get_available_actions()
                while True:
                    try:
                        move_input = input(f"Your turn (O). Enter position {list(available_actions)}: ")
                        move = int(move_input)
                        if move in available_actions:
                            break
                        else:
                            print(f"Invalid move. Choose from {list(available_actions)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number between 0 and 8.")
                
                state, reward, is_done, info = env.step(move)

            env.render()
            
            if is_done:
                if env.winner == 1:
                    print("GAME OVER: Agent (X) wins!")
                elif env.winner == 2:
                    print("GAME OVER: Congratulations! You (O) win!")
                else:
                    print("GAME OVER: It's a draw!")
                break

        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    play_game()
