import gym
import torch
import numpy as np
import pickle
from agent import DQNAgent
from buffer import BUF
from config import CFG
# import data
# train model
# evaluate model



env = gym.make("BipedalWalker-v3",hardcore=True, render_mode='human')

iterations = 1000

#print(env.observation_space.shape[0], env.action_space.shape[0])

agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])

terminated = True
for _ in range(iterations):

    if terminated :
        obs_old, info = env.reset()

    action = agent.get(obs_old, env.action_space)

    obs_new, reward, terminated, truncated, info = env.step(action)

    BUF.set((obs_old, action, reward, obs_new))

    try :
        old_list, act_list, rwd_list, new_list = BUF.get()
        for i in range(CFG.batch_size):
            agent.set(old_list[i], act_list[i], rwd_list[i], new_list[i])
            print(f'training {i} of step{_}completed')

    except :
        agent.set(obs_old, action, reward, obs_new)

    obs_old = obs_new

    #print(BUF.len())





'''
with open('01_Bidepal_Walker/data/data.pickle', 'wb') as f:
    for i in range(iterations):
        print(f"{i}/{iterations} ie {round(i/iterations,2)*100}%")
        observation, info = env.reset()
        for _ in range(100):
            state = observation
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = observation
            tuple = (state, action, next_state, reward)


            pickle.dump(tuple, f)



objects = []
with (open('01_Bidepal_Walker/data/data.pickle', "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(len(objects))

rewards = [i[3] for i in objects]
print(max(rewards))
'''
