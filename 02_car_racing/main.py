# Import for environment
import gym
import torch
import numpy as np
import pickle
import time

# Import for videorecording
import os
import moviepy
from gym.utils.save_video import save_video

# Import Classes
from agent import DQNAgent
from buffer import BUF
from config import CFG
import wrapper




# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available and being used")
# else:
#     device = torch.device("cpu")
#     print("GPU is not available, using CPU instead")


env = gym.make("CarRacing-v2", continuous=False, render_mode='human')

# env.reset()
# for i in range(200):
#     action = env.action_space.sample()
#     env.step(3.5)
#     print(action)



# # 1. Initializing agent
print('initializing agent')
agent = DQNAgent(3, env.action_space.shape)
print(type(agent).__name__)

print('______')

# 2.1 Start training returning best rewards
print('training agent')
best_reward=agent.training(100)
# best_reward=agent.training_and_record()

# # # 3. Saving agent (falcultatif)
save = ''
path = os.path.join(os.path.dirname(__file__), f"./data/")

while save != 'y' and save!='n':
    save = input("Souhaitez vous enregistrer l'agent ? (y/n)")
    if save == 'y':
        print('saving agent')
        path = os.path.join(os.path.dirname(__file__), f"./data/")
        agent.save(path, best_reward)



# # 2.2 Load Agent

# print ('loading agent')
# agent.load('01_Bidepal_Walker/data/saved_model_DQNAgent__10399.58rw__1000episodes__200steps.pt')




print('______')


# 1. Evaluation
print('evaluating model (opt:recording)')
input('evaluate')
number_of_steps = 300
recording = False
agent.evaluate(number_of_steps,recording)
