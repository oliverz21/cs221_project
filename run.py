import pandas as pd
from dqn import DQNAgent, SARSAAgent, sureFire
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_excel('eur_data.xlsx')).to_numpy()
n = len(data)
dataTrain, dataTest = data[: 3000], data[3000:]
dataTrain = np.hstack((dataTrain, np.linspace(1, 0, len(dataTrain)).reshape(len(dataTrain), 1)))
dataTest = np.hstack((dataTest, np.linspace(1, 0, len(dataTest)).reshape(len(dataTest), 1)))
state_size = 15
# actions
actions_upperLimit = [0, 1, 2, 3, 4, 5]
actions_firstBuyOrSell = [1, -1]
actions_k = [i / 100.0 for i in range(1, 9, 1)]
action_size = len(actions_upperLimit) * len(actions_firstBuyOrSell) * len(actions_k)

agent = SARSAAgent(state_size, action_size)
# agent = DQNAgent(state_size, action_size)
EPISODES = agent.episodes
done = False
batch_size = 32


def chooseActions(a):
    n_k = len(actions_k)
    n_firstBuyOrSell = len(actions_firstBuyOrSell)
    a1 = a // (n_firstBuyOrSell * n_k)
    a2 = (a % (n_firstBuyOrSell * n_k)) // n_k
    a3 = (a % (n_firstBuyOrSell * n_k)) % n_k
    return actions_upperLimit[a1], actions_firstBuyOrSell[a2], actions_k[a3]


x_episode = []
y_profit = []

for e in range(EPISODES + 1):
    # train
    # randomly pick the start of an episode within the first 300 days
    t = random.randrange(300)
    state = dataTrain[t]

    while (True):
        action = agent.act(state)
        upperLimitBuyin, firstBuyOrSell, k = chooseActions(action)
        sf = sureFire(dataTrain[t][0], upperLimitBuyin, firstBuyOrSell, k, t)
        t, reward, profit = sf.runSF(dataTrain.T[0])
        next_state = dataTrain[t]
        # add to experience memory
        agent.remember(state, action, reward, next_state)
        state = next_state
        sys.stdout.write("\rtrain {}/{}".format(e, EPISODES))
        if t == len(dataTrain) - 1:
            break

    # experience replay
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, e)

    print ("\n")
    # test
    t = 0
    profitSum = 0
    while (True):
        state = dataTest[t]
        state = state.reshape(len(state), 1).T
        act_values = agent.model.predict(state)
        action = np.argmax(act_values[0])
        upperLimitBuyin, firstBuyOrSell, k = chooseActions(action)
        sf = sureFire(dataTest[t][0], upperLimitBuyin, firstBuyOrSell, k, t)
        tt, reward, profit = sf.runSF(dataTest.T[0])
        a1, a2, a3 = chooseActions(action)
        print ("t_start: " + str(t) + ", profit: " + str(profit) + " ,t_end: " + str(tt))
        print ("a1: " + str(a1) + ", a2: " + str(a2) + " ,a3: " + str(a3))
        t = tt
        profitSum += profit
        if t == len(dataTest) - 1:
            break
    # ignore the last transaction because in most cases it is an incomplete sure-fire
    print ("Episode: " + str(e) + ", profit: " + str(profitSum - profit))
    x_episode.append(e)
    y_profit.append(profitSum - profit)

plt.plot(x_episode, y_profit)
plt.xlabel('Episode #')
plt.ylabel('Profit')
plt.show()

plt.plot([x_episode[i] for i in range(20, 1980, 10)], [sum(y_profit[i - 20:i + 20]) / 40 for i in range(20, 1980, 10)])
plt.xlabel('Episode #')
plt.ylabel('Average Profit')
plt.show()