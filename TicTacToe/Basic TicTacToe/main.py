from game import TicTacToe

def play_game():
    """
    Handles the gameplay loop for Tic-Tac-Toe.
    """
    # Initialize the game
    tictactoe = TicTacToe()
    player = 'X'  # Player 'X' starts
    state = tictactoe.get_initial_state()

    while True:
        # Display the board
        print(tictactoe.render_board(state))
        
        # Get valid moves
        valid_moves = tictactoe.get_valid_moves(state)
        print("Valid moves:", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        
        # Player inputs their move
        try:
            action = int(input(f"Player {player} (Enter a valid position): "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 8.")
            continue

        # Validate the action
        if action < 0 or action >= tictactoe.action_size or valid_moves[action] == 0:
            print("Invalid move. Please try again.")
            continue

        # Update the state
        state = tictactoe.get_next_state(state, action, player)
        
        # Check if the game is over
        value, is_terminal = tictactoe.get_value_and_terminated(state, action)
        if is_terminal:
            print(tictactoe.render_board(state))
            if value == 1:
                print(f"Player {player} wins!")
            else:
                print("It's a draw!")
            break

        # Switch to the opponent
        player = tictactoe.get_opponent(player)

if __name__ == "__main__":
    play_game()
