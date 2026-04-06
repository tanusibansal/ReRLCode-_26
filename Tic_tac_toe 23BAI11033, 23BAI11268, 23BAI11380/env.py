import numpy as np

class TicTacToeEnv:
    """
    A simple Tic-Tac-Toe environment for Reinforcement Learning.
    Players are 1 (X) and 2 (O).
    Board is represented as a 1D array of 9 elements.
    """
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, 2 for O
        self.winner = None
        self.is_done = False

    def reset(self):
        """Resets the environment for a new game."""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.winner = None
        self.is_done = False
        return self.get_state()

    def get_state(self):
        """Returns the current state of the board as a tuple."""
        return tuple(self.board)

    def get_available_actions(self):
        """Returns a list of indices where the board is empty."""
        return np.where(self.board == 0)[0]

    def step(self, action):
        """
        Takes an action and returns (next_state, reward, is_done, info).
        Reward:
            1: Player wins
            0.5: Draw
            0: Game continues
            -10: Illegal move (though the agent should ideally pick valid moves)
        """
        if self.board[action] != 0:
            # This should ideally be handled by the agent choosing only valid actions
            return self.get_state(), -10, True, {"msg": "Illegal move"}

        self.board[action] = self.current_player
        
        if self.check_winner(self.current_player):
            self.winner = self.current_player
            self.is_done = True
            return self.get_state(), 1, True, {"msg": "Win"}

        if len(self.get_available_actions()) == 0:
            self.is_done = True
            return self.get_state(), 0.5, True, {"msg": "Draw"}

        self.current_player = 3 - self.current_player # Switch player: 1 -> 2, 2 -> 1
        return self.get_state(), 0, False, {"msg": "Continue"}

    def check_winner(self, player):
        """Checks if the given player has won the game."""
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # cols
            (0, 4, 8), (2, 4, 6)             # diagonals
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return True
        return False

    def render(self):
        """Prints the board to the console."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n")
        for i in range(0, 9, 3):
            row = self.board[i:i+3]
            print(f" {symbols[row[0]]} | {symbols[row[1]]} | {symbols[row[2]]} ")
            if i < 6:
                print("-----------")
        print("\n")
