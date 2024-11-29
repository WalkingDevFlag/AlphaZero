import numpy as np

class TicTacToe:
    """
    A class to represent the Tic-Tac-Toe game mechanics.
    """

    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count  # Total number of possible actions (9 for 3x3)

    def get_initial_state(self):
        """
        Returns the initial empty game board.
        """
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state, action, player):
        """
        Updates the board state with the player's move.
        Args:
            state: Current state of the board.
            action: The position (0-8) chosen by the player.
            player: Current player ('X' or 'O').

        Returns:
            Updated board state after the move.
        """
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = 1 if player == 'X' else -1
        return state

    def get_valid_moves(self, state):
        """
        Returns an array of valid moves (1 for valid, 0 for invalid).
        Args:
            state: Current state of the board.

        Returns:
            A binary array indicating valid moves.
        """
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        """
        Checks if the last move results in a win.
        Args:
            state: Current state of the board.
            action: Last action taken by the player.

        Returns:
            True if the player wins, False otherwise.
        """
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        """
        Evaluates the board state to determine if the game is over and the winner.
        Args:
            state: Current state of the board.
            action: Last action taken.

        Returns:
            (value, terminated):
                value: 1 if the current player wins, 0 for a draw.
                terminated: True if the game has ended, False otherwise.
        """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        """
        Returns the opponent's player value.
        Args:
            player: Current player ('X' or 'O').

        Returns:
            'O' if player is 'X', and 'X' if player is 'O'.
        """
        return 'O' if player == 'X' else 'X'

    def render_board(self, state):
        """
        Renders the board in a user-friendly format.
        Args:
            state: Current state of the board.

        Returns:
            A string representation of the board with 'X', 'O', and '_'.
        """
        board = state.copy()
        render_map = {0: '_', 1: 'X', -1: 'O'}
        return "\n".join(["[" + " ".join(render_map[cell] for cell in row) + "]" for row in board])
