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

    # Start episode
    start_time=time.time()
    terminated = True
    r = 0
    for step in range(max_steps):

        # Reinitializing
        if terminated :
            obs_old, info = env.reset()

        # Get action from agent and give it to environment
        action = agent.get(obs_old, env.action_space)
        obs_new, reward, terminated, truncated, info = env.step(action)
        r+=reward
        # Storing step into buffer
        BUF.set((obs_old, action, reward, obs_new, terminated))


        # Training on a batch of the buffer if large enough otherwise only on this step
        # try :
        #     old_list, act_list, rwd_list, new_list, new_terminated = BUF.get()
        #     for i in range(CFG.batch_size):
        #         agent.set(old_list[i], act_list[i], rwd_list[i], new_list[i], new_terminated[i])
        # except :
        #     agent.set(obs_old, action, reward, obs_new, terminated)

        obs_old = obs_new

    old_list, act_list, rwd_list, new_list, new_terminated = BUF.get()

    agent.set(old_list, act_list, rwd_list, new_list, new_terminated)


    end_time = time.time()
    total_rewards.append(r)

    # Completion status
    percent = round((episode+1)/episodes*100,2)
    duration = round(end_time - start_time, 2) #in sec
    remaining_est = (episodes-episode) * duration

    if episode%100 ==0:
        print(f'{percent} % done | duration : {duration} sec | estim left : {remaining_est} sec')

#Computing best rewards
best_reward = max(total_rewards)

path = os.path.join(os.path.dirname(__file__), f"./data/")
agent.save(path, best_reward)

agent.load('01_Bidepal_Walker/data/saved_model_DQNAgent__4.27rw__300000episodes__400steps.pt')
# Reinitializing environment with render
env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')

terminated = False

#evaluation/vizualization over 1000 steps
input("Press enter to see validation ")
obs_old, info = env.reset()
episode_index = 0
step_starting_index =0

for step in range(1000):
    if terminated :
        # save_video(
        #     env.render(),
        #     "videos-bidepal",
        #     fps=env.metadata["render_fps"],
        #     step_starting_index=step_starting_index,
        #     episode_index=episode_index,
        #     name_prefix = f"{type(agent).__name__}__{CFG.episodes}episodes__{CFG.max_steps}steps"
        # )
        episode_index += 1
        step_starting_index = step+1

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
