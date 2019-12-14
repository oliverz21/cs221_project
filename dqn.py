import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=30000)
        self.gamma = 0.5    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.episodes = 4000


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # get action
    def act(self, state):
        # select random action with prob=epsilon else action=maxQ
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(len(state), 1).T
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, t):
        # sample random transitions
        for _ in range(int(10 // self.epsilon)-9):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state in minibatch:
                state = state.reshape(len(state), 1).T
                next_state = next_state.reshape(len(next_state), 1).T
                Q_next = self.model.predict(next_state)[0]
                alpha = self.episodes / float(self.episodes + t)
                target = reward + self.gamma * np.amax(Q_next)
                target_f = self.model.predict(state)
                target_f[0][action] = target_f[0][action] + alpha * (target - target_f[0][action])
                # train network
                self.model.fit(state, target_f, epochs=1, verbose=0)


class SARSAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=30000)
        self.gamma = 0.5    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.episodes = 2000


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # get action
    def act(self, state):
        # select random action with prob=epsilon else action=maxQ
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(len(state), 1).T
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, t):
        alpha = self.episodes / float(self.episodes + t)
        # sample random transitions
        for _ in range(int(10 // self.epsilon)-9):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state in minibatch:
                state = state.reshape(len(state), 1).T
                next_state = next_state.reshape(len(next_state), 1).T
                Q_next = self.model.predict(next_state)[0]
                target = reward + self.gamma * random.choice(Q_next)
                target_f = self.model.predict(state)
                target_f[0][action] = target_f[0][action] + alpha * (target - target_f[0][action])
                # train network
                self.model.fit(state, target_f, epochs=1, verbose=0)

# buy:   1
# sell: -1
class sureFire:
    def __init__(self, unitPrice, upperLimitBuyin, firstBuyOrSell, k, currDay):
        self.unit = unitPrice
        self.k = k
        self.t = currDay
        self.upperLimitBuyin = upperLimitBuyin
        self.firstBuyOrSell = firstBuyOrSell
        self.lastBuyOrSell = firstBuyOrSell
        self.numBuyin = 1 if firstBuyOrSell is 1 else 0
        self.hold = 1 if firstBuyOrSell is 1 else -1
        self.cash = -unitPrice if firstBuyOrSell is 1 else unitPrice
        self.stopGain = unitPrice + k if firstBuyOrSell is 1 else unitPrice - k
        self.stopLoss = unitPrice - 2*k if firstBuyOrSell is 1 else unitPrice + 2*k
        self.backhand = unitPrice - k if firstBuyOrSell is 1 else unitPrice + k

    def reward(self, price):
        # reward = profit * discount
        profit = self.hold * price + self.cash
        discount = 1 - 0.1 * (self.numBuyin-1)
        return profit * discount

    def profit(self, price):
        return self.hold * price + self.cash

    def runSF(self, data):
        t = self.t
        while (t < len(data) - 1):

            t = t + 1
            if self.upperLimitBuyin == 0:
                return t, 0, 0
            # if last transaction is buy
            if self.lastBuyOrSell is 1:
                # sell all holds if reach stop gain
                if data[t] > self.stopGain:
                    return t, self.reward(data[t]), self.profit(data[t])
                # sell 3 times current hold if reach backhand
                if data[t] < self.backhand:
                    self.cash += data[t] * self.hold * 3
                    self.hold = -self.hold * 2
                    self.lastBuyOrSell = -1
                    self.stopGain, self.stopLoss = self.stopLoss, self.stopGain
                    self.backhand = self.backhand + self.k

            # if last transaction is sell
            if self.lastBuyOrSell is -1:
                # return all owed holds if reach stop gain
                if data[t] < self.stopGain:
                    return t, self.reward(data[t]), self.profit(data[t])
                # if reach backhand
                if data[t] > self.backhand:
                    # if reach upper limit of buyins
                    if self.numBuyin >= self.upperLimitBuyin:
                        # print "MEET UPPER LIMIT FOR BUYINS"
                        return t, self.profit(data[t]), self.profit(data[t])
                    # buy 3 times current owed
                    self.cash -= data[t] * (-self.hold) * 3
                    self.hold = -self.hold * 2
                    self.lastBuyOrSell = 1
                    self.stopGain, self.stopLoss = self.stopLoss, self.stopGain
                    self.backhand = self.backhand - self.k
                    self.numBuyin += 1

        return t, self.profit(data[t]), self.profit(data[t])