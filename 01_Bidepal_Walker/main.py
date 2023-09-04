# Import for environment
import gym
import torch
import numpy as np
import pickle
import time

# Import for videorecording
import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy
from gym.utils.save_video import save_video

# Import Classes
from agent import DQNAgent
from buffer import BUF
from config import CFG
import wrapper

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
print(type(agent).__name__)

# Start training
total_rewards = []
for episode in range(episodes):

    if episode %1000 == 0:
        env = gym.make("BipedalWalker-v3",hardcore=False)#, render_mode='human')
    else :
        env = gym.make("BipedalWalker-v3",hardcore=False)


    # Start episode
    start_time=time.time()

    r = 0

    obs_old, _ = env.reset()

    env = wrapper.RewardWrapper(env)


    for step in range(max_steps):

        # Get action from agent and give it to environment
        action = agent.get(obs_old, env.action_space)
        obs_new, reward, terminated, truncated, _ = env.step(action)
        r += reward

        done = terminated or truncated

        # if reward < 0 :
        #     reward = 0
        # else :
        #     reward *= 100

        # Storing step into buffer
        BUF.set((obs_old, action, reward, obs_new, done))

        # Training agent if buffer is full (no need to clear it because size)
        if agent.time_step % CFG.buffer_size == 0 :
            old_list, act_list, rwd_list, new_list, new_terminated = BUF.get()
            agent.set(old_list, act_list, rwd_list, new_list, new_terminated)


        # Updating target after sync_every steps
        if agent.time_step % CFG.sync_every == 0:
            agent.target.load_state_dict(agent.net.state_dict())
            print('Target updated')

        obs_old = obs_new

        # Reinitializing
        if done:
            break

    end_time = time.time()
    total_rewards.append(r)

    # Completion status
    percent = round((episode+1)/episodes*100,2)
    duration = round(end_time - start_time, 2) #in sec
    remaining_est = (episodes-episode) * duration

    if episode%100 ==0:
        print(f'{percent} % done | duration : {duration} sec | estim left : {remaining_est} sec')
        print(agent.exploration_rate)

        env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')
        obs_old, info = env.reset()

        for step in range(500):
            action = agent.get(obs_old, env.action_space, evaluating=True)

            obs_new, reward, terminated, truncated, _ = env.step(action)

            obs_old = obs_new
            if terminated:
                break



#Computing best rewards
best_reward = max(total_rewards)

path = os.path.join(os.path.dirname(__file__), f"./data/")
#agent.save(path, best_reward)

#agent.load('01_Bidepal_Walker/data/saved_model_DQNAgent__4.27rw__300000episodes__400steps.pt')
# Reinitializing environment with render
env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')

terminated = False

#evaluation/vizualization over 1000 steps
input("Press enter to see validation ")
obs_old, info = env.reset()
episode_index = 0
step_starting_index =0

for step in range(1000):
    action = agent.get(obs_old, env.action_space, evaluating=True)

    obs_new, reward, terminated, truncated, _ = env.step(action)

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
