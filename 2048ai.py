from game import Game
import random

import math

         
    

if __name__ == '__main__':
    env = Game()
    while True:
        print(env.string() + "\n")
        print env.actions()
        action = random.choice(env.actions())
        
        s, r, done = env.step(action)
        env.b = s
        #print done
        if done:
            m = max(x for row in s for x in row)
            print("env over...best was %s" %m)
            print(env.string())
            break 