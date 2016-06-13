from game import Game
import random

import math

         
def aiplay(b):
    """
    Runs the board playing the move that determined
    by aimove.
    """
    while True:
        print(board.string(b) + "\n")
        action = random.choice(board.actions(b))
        
        if action == 0 : b = board.left(b)
        if action == 1 : b = board.right(b)
        if action == 2 : b = board.up(b)
        if action == 3 : b = board.down(b)
        b = board.spawn(b, 1)
        if board.over(b):
            m = max(x for row in b for x in row)
            print("board over...best was %s" %m)
            print(board.string(b))
            break 

if __name__ == '__main__':
    board = Game()
    #board.b = board.spawn(board.b, 2)
    aiplay(board.b)