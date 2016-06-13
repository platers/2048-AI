import random
from itertools import *

class Game:
    def __init__(self):
        """ Initialize a 2048 self
            NOTE: self does not know how to "play itself". Think of "self"
            as representing a starting board configuration with the ability
            to advance and play 2048.
        """
        board = [[0]*4 for i in range(4)]
        self.b = self.spawn(board, 2)

    def actions(self, b):
        """ Generate the subsequent board after moving """
        array = []
        def moved(b, t):
            return any(x != y for x, y in zip(b, t)) 
            
        t = self.left(b)
        if moved(b, t):
            array.append(0)
        t = self.right(b)
        if moved(b, t):
            array.append(1)
        t = self.up(b)
        if moved(b, t):
            array.append(2)
        t = self.down(b)
        if moved(b, t):
            array.append(3)
        return array

    def over(self, b):
        """ Return whether or not a board is playable
        """
        def inner(b):
            for row in b:
                for x, y in zip(row[:-1], row[1:]):
                    if x == y or x == 0 or y == 0:
                        return True
            return False
        return not inner(b) and not inner(zip(*b))

    def string(self, b):
        """ String to pretty print the board in matrix form """
        
        return '\n'.join([''.join(['{:8}'.format(item) for item in row])
                                for row in b])

    def spawn(self, b, k=1):
        """ Add k random tiles to the board.
            Chance of 2 is 90%; chance of 4 is 10% """
        rows, cols = list(range(4)), list(range(4))
        random.shuffle(rows)
        random.shuffle(cols)
        
        copy  = [[x for x in row] for row in b]
        dist  = [2]*9 + [4]
        count = 0
        for i,j in product(rows, rows):
            if copy[i][j] != 0: continue
            
            copy[i][j] = random.sample(dist, 1)[0]
            count += 1
            if count == k : return copy
        raise Exception("shouldn't get here")
        
    def left(self, b):
        """ Returns a left merged board
        >>> self.left(test)
        [[2, 8, 0, 0], [2, 8, 4, 0], [4, 0, 0, 0], [4, 4, 0, 0]]
        """

        return self.merge(b)

    def right(self, b):
        """ Returns a right merged board
        >>> self.right(test)
        [[0, 0, 2, 8], [0, 2, 4, 8], [0, 0, 0, 4], [0, 0, 4, 4]]
        """

        def reverse(x):
            return list(reversed(x))
        
        t = map(reverse, iter(b))
        return [reverse(x) for x in self.merge(t)]

    def up(self, b):
        """ Returns an upward merged board
            NOTE: zip(*t) is transpose
            
        >>> self.up(test) 
        [[4, 8, 4, 8], [4, 2, 0, 2], [0, 0, 0, 4], [0, 0, 0, 0]]
        """

        t = self.left(zip(*b))
        return [list(x) for x in zip(*t)]

    def down(self, b):
        """ Returns an downward merged board
            NOTE: zip(*t) is transpose
        >>> self.down(test)
        [[0, 0, 0, 0], [0, 0, 0, 8], [4, 8, 0, 2], [4, 2, 4, 4]]
        """
        
        t = self.right(zip(*b))
        return [list(x) for x in zip(*t)]

    def merge(self, b):
        """ Returns a left merged board """
        
        def inner(row, a):
            """
            Helper for merge. If we're finished with the list,
            nothing to do; return the accumulator. Otherwise
            if we have more than one element, combine results of first
            with right if they match; skip over right and continue merge
            """
            
            if not row:
                return a
            x = row[0]
            if len(row) == 1:
                return inner(row[1:], a + [x])
            return inner(row[2:], a + [2*x]) if x == row[1] else inner(row[1:], a + [x])

        ret = []
        for row in b:
            merged = inner([x for x in row if x != 0], [])
            merged = merged + [0]*(len(row)-len(merged))
            ret.append(merged)
        return ret