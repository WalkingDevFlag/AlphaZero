from game import ConnectFour
from MCTS import MCTS
import numpy as np

# Initialize game and MCTS
game = ConnectFour()
args = {
    'C': 2,
    'num_searches': 100
}
mcts = MCTS(game, args)

# Start game
state = game.get_initial_state()
player = 1

while True:
    game.render_board(state)

    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("Valid moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"Player {player}, choose an action: "))

        if valid_moves[action] == 0:
            print("Invalid action!")
            continue
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        game.render_board(state)

        if value == 1:
            print(f"Player {player} won!")
        else:
            print("It's a draw!")
        break
        
    player = game.get_opponent(player)
