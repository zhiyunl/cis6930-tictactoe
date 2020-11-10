'''
Author: Zhiyun Ling
Date: 2020-10-08 16:10:44
LastEditTime: 2020-10-08 16:31:15
'''
"""
TicTacToe Game
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import time


def check_winner(state, p_id):
    if (state[0] == state[1] == state[2] == p_id) or (state[3] == state[4] == state[5] == p_id) \
            or (state[6] == state[7] == state[8] == p_id) or (state[0] == state[3] == state[6] == p_id) \
            or (state[1] == state[4] == state[7] == p_id) or (state[2] == state[5] == state[8] == p_id) \
            or (state[0] == state[4] == state[8] == p_id) or (state[2] == state[4] == state[6] == p_id):
        return p_id
    else:
        return 0


"""
Loading the multilayer Perceptron as mlp_tictac from the dill file mlp_tictac.pkl using the pickle package
"""
with open('mlp_tictac.pkl', 'rb') as f:
    mlp_tictac = pickle.load(f)

### making the board display and the state of the board

boardDisplay = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9'])

boardState = np.zeros(9)


def print_board():
    print('\n', boardDisplay[0], '|', boardDisplay[1], '|', boardDisplay[2], '\n-----------\n', boardDisplay[3], '|',
          boardDisplay[4], '|', boardDisplay[5], '\n-----------\n', boardDisplay[6], '|', boardDisplay[7], '|',
          boardDisplay[8])


def main():
    ### Player starts the game

    for i in range(4):
        print('It\'s your turn!')
        print_board()
        ## first player move
        p_pos = int(input('\nChoose your next move x \n: 1 to 9\n'))
        if 1 <= p_pos <= 9:
            if boardState[p_pos - 1] != 0:
                print("Already Taken!")
                return
            boardDisplay[p_pos - 1] = 'X'
            boardState[p_pos - 1] = 1
            print_board()
            if check_winner(boardState, p_id=1) == 1:
                print('\n****************\nCongratulations! You Win!!\n****************\n')
                return
            ## now computers move
            print('Now my turn')
            c_pos = mlp_tictac.predict(boardState.reshape(1, -1))
            # find the next possible position under largest score
            for j in range(len(c_pos)):
                c1 = np.argmax(c_pos)
                if boardState[c1] == 0:
                    boardState[c1] = -1
                    boardDisplay[c1] = 'O'
                    break
                else:
                    c_pos[c1] = 0
                    continue

            time.sleep(1)
            print('I placed a \'O\' on the board at index', c1 + 1)
            print_board()
            if check_winner(boardState, -1) == -1:
                print('\n***********************\nI Win!!\n***********************\n')

                return
        else:
            ##index out of bound
            print(color.BOLD + '\nGAME ENDED coz of Rules Violation' + color.END)
            print('Your move is out of bound')
            return
    print('\n****************\nITS A TIE!!\n****************\n')


if __name__ == "__main__":
    main()
