import torch
import numpy as np
from game import TicTacToe
from MCTS import MCTS
from ResNet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize game
tictactoe = TicTacToe()

# Load model
model = ResNet(tictactoe, 4, 64, device=device)
model.load_state_dict(torch.load("E:\Random Python Scripts\kaggle chess\TicTacToe\AlphaZero TicTacToe\model_2.pt", map_location=device))
model.eval()

args = {
    'C': 2,
    'num_searches': 1000,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

mcts = MCTS(tictactoe, args, model)
state = tictactoe.get_initial_state()
player = 1

# Helper function to display the board
def display_board(state):
    symbols = {0: '_', 1: 'X', -1: 'O'}
    board = "\n".join([" ".join([symbols[cell] for cell in row]) for row in state])
    print(board)

while True:
    display_board(state)
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("Valid moves:", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"Player {'X'}, choose an action: "))
        if valid_moves[action] == 0:
            print("Invalid action!")
            continue
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        print(f"AI chooses action: {action}")
    
    state = tictactoe.get_next_state(state, action, player)
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    if is_terminal:
        display_board(state)
        if value == 1:
            print(f"Player {'X' if player == 1 else 'O'} wins!")
        else:
            print("It's a draw!")
        break
    player = tictactoe.get_opponent(player)



'''
import torch
import numpy as np
from game import TicTacToe
from MCTS import MCTS
from ResNet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize game
tictactoe = TicTacToe()

# Load model
model = ResNet(tictactoe, 4, 64, device=device)
model.load_state_dict(torch.load("E:\Random Python Scripts\kaggle chess\TicTacToe\AlphaZero TicTacToe\model_2.pt", map_location=device))
model.eval()

args = {
    'C': 2,
    'num_searches': 1000,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

mcts = MCTS(tictactoe, args, model)
state = tictactoe.get_initial_state()
player = 1

while True:
    print(state)
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("Valid moves:", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"Player {player}, choose an action: "))
        if valid_moves[action] == 0:
            print("Invalid action!")
            continue
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
    
    state = tictactoe.get_next_state(state, action, player)
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    if is_terminal:
        print(state)
        if value == 1:
            print(f"Player {player} wins!")
        else:
            print("It's a draw!")
        break
    player = tictactoe.get_opponent(player)
'''