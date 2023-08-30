import gym
import torch
import numpy as np
import pickle
import time

from agent import DQNAgent
from buffer import BUF
from config import CFG

# import data
# train model
# evaluate model


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available and being used")
# else:
#     device = torch.device("cpu")
#     print("GPU is not available, using CPU instead")


env = gym.make("BipedalWalker-v3",hardcore=False)

episodes = CFG.episodes
max_steps = CFG.max_steps


# Initializing agent
agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])


# Start training
for episode in range(episodes):

    # Start episode
    start_time=time.time()
    terminated = True
    for step in range(max_steps):

        # Reinitializing
        if terminated :
            obs_old, info = env.reset()

        # Get action from agent and give it to environment
        action = agent.get(obs_old, env.action_space)
        obs_new, reward, terminated, truncated, info = env.step(action)

        # Storing step into buffer
        BUF.set((obs_old, action, reward, obs_new))


        # Training on a batch of the buffer if large enough otherwise only on this step
        try :
            old_list, act_list, rwd_list, new_list = BUF.get()
            for i in range(CFG.batch_size):
                agent.set(old_list[i], act_list[i], rwd_list[i], new_list[i])
        except :
            agent.set(obs_old, action, reward, obs_new)

        obs_old = obs_new

    end_time = time.time()

    # Completion status
    percent = round((episode+1)/episodes*100,2)
    duration = round(end_time - start_time, 2) #in sec
    remaining_est = 1/(percent/100) * duration
    print(f'{percent} % done | duration : {duration} sec | estim left : {remaining_est}')

# path = os.path.join(os.path.dirname(__file__), f"../data/")
# agent.save(path)

# Reinitializing environment with render
env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')

terminated = True

#evaluation/vizualization over 1000 steps
input("Press enter to see validation ")
for step in range(1000):

    if terminated :
            obs_old, info = env.reset()

    action = agent.get(obs_old, env.action_space, evaluating=True)

    obs_new, reward, terminated, truncated, info = env.step(action)

    obs_old = obs_new



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
