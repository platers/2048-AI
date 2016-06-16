from game import Game
import random
import numpy as np
import math

e = 1
iterations = 200
max_memory = 1000000
hidden_size = 30
num_actions = 4
input_size = 16
batch_size = 100
totalSteps = 0
learningRate = 0.000025
learnStart = 1000

class ExperienceReplay(object):
    def __init__(self, max_memory, discount):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=batch_size):
        indices = random.sample(np.arange(len(self.memory)), min(batch_size,len(self.memory)) )
        miniBatch = []
        for index in indices:
            miniBatch.append(self.memory[index])
        return miniBatch

from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Activation
class DeepQ:
    def createModel(self, activationType, learningRate):
            model = Sequential()
            model.add(Dense(100, input_shape=(input_size, ), init='lecun_uniform'))
            model.add(Activation(activationType))
            layerSize = 100
            model.add(Dense(layerSize, init='lecun_uniform'))
            model.add(Activation(activationType))
            model.add(Dense(layerSize, init='lecun_uniform'))
            model.add(Activation(activationType))
            model.add(Dense(num_actions, init='lecun_uniform'))
            model.add(Activation("linear"))
            optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
            model.compile(loss="mse", optimizer=optimizer)
            #print model.summary()
            self.model = model
    def getAction(self, state):
        qValues = self.model.predict(state.reshape(1,len(state)))[0]
        return np.argmax(qValues)
    def trainModel(self, batch, discount):
        X_batch = np.empty((0, input_size), dtype = np.float64)
        Y_batch = np.empty((0, num_actions), dtype = np.float64)
        for sample in batch:
            state = sample[0][0]
            action = sample[0][1]
            reward = sample[0][2]
            newState = sample[0][3]
            isFinal = sample[1]
            qValues = self.model.predict(state.reshape(1,len(state)))[0]
            qValuesNewState = self.model.predict(newState.reshape(1,len(newState)))[0]
            targetValue = reward + discount * max(qValuesNewState)

            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
            Y_sample = qValues.copy()
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
                X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward]*num_actions]), axis=0)
        return self.model.train_on_batch(X_batch, Y_batch)

if __name__ == '__main__':
    env = Game()
    exp_replay = ExperienceReplay(max_memory, 0.99)
    DQN = DeepQ()
    DQN.createModel('relu', learningRate)
    for i_episode in range(iterations):
        env = Game()
        loss = 0
        t = 0
        while True:
            #print(env.string() + "\n")
            #print env.actions()
            e *= 0.9995
            if e < 0.05:
                e = 0.05
            s = env.state()
            action = random.choice(env.actions())
            if(random.random() <= e):
                action = random.choice(env.actions())
            else:
                a = DQN.getAction(s)
                if a in env.actions():
                    action = a
            ss, ss_formated, r, done = env.step(action)
            env.b = ss
            t += 1
            totalSteps += 1
            if done:
                r -= 200
            exp_replay.remember([s, action, r, ss_formated], done)
            if totalSteps > learnStart:
                loss += DQN.trainModel(exp_replay.get_batch(batch_size), 0.99)
            if done:
                m = max(ss_formated)
                print("Iteration {} complete with time {} and loss {} e = {}".format(i_episode, t, loss / t, e))
                #print(env.string())
                break 