import numpy as np
from game import ConnectFour
from ResNet import ResNet
from MCTS import MCTS

game = ConnectFour()
player = 1

args = {
    'C': 2,
    'num_searches': 100
}

model = ResNet(game, 4, 64)
model.eval()

mcts = MCTS(game, args, model)

state = game.get_initial_state()

while True:
    game.render_board(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("Valid moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input("Enter your move: "))
    else:
        action_probs = mcts.search(state)
        action = np.argmax(action_probs)
    
    state = game.get_next_state(state, action, player)
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        if value == 1:
            print(f"Player {player} wins!")
        else:
            print("It's a draw!")
        break
    
    #state = game.change_perspective(state, player=-1)
    player = game.get_opponent(player)
