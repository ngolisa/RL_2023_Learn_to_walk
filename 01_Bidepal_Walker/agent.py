"""
Agent module.
"""

import random
import torch
import torch.nn
import model
from config import CFG
from torch.nn import functional as F
import numpy as np
import gym
from wrapper import RewardWrapper
import time
from buffer import BUF
from gym.utils.save_video import save_video
import datetime

today =  datetime.datetime.now()



class Agent:
    """
    A learning agent parent class.
    """

    def __init__(self):
        pass

    def set(self):
        """
        Make the agent learn from a (s, a, r, s') tuple.
        """
        raise NotImplementedError

    def get(self):
        """
        Request a next action from the agent.
        """
        raise NotImplementedError

    def save(self, path: str, best_reward):
        """
        Save the agent's model to disk.
        """
        torch.save(self.net.state_dict(), f"{path}saved_model_{type(self).__name__}__{round(best_reward,2)}rw__{CFG.episodes}episodes__{CFG.max_steps}steps.pt")


    def load(self, path: str):
        """
        Load the agent's weights from disk.
        """
        data = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(data)


class RandomAgent(Agent):
    """
    A random playing agent class.
    """

    def set(self, obs_old, act, rwd, obs_new):
        """
        A random agent doesn't learn.
        """
        return

    def get(self, obs_new, act_space):
        """
        Simply return a random action.
        """
        return act_space.sample()

class DQNAgent(Agent):
    """
    A basic DQN agent
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        """
        Initializing model and parameters
        """
        self.net = model.DQN(obs_size, action_size)
        self.opt = torch.optim.Adam(params=self.net.parameters(), lr=CFG.learning_rate)
        self.target = model.DQN(obs_size, action_size)
        self.target.load_state_dict(self.net.state_dict())
        self.time_step = 0
        self.exploration_rate = CFG.exploration_rate
        self.exploration_rate_min = CFG.exploration_rate_min
        self.exploration_rate_decay = CFG.exploration_rate_decay


    def get(self, obs_new, act_space, evaluating=False):
        """
        Next action selection
        """
        # Evaluating agent means rely on learnt behavior (no randomness)
        if evaluating:
            self.exploration_rate=0


        # Explore : Returns a random action with p = exploration_rate
        if random.uniform(0,1) < self.exploration_rate:
            what_to_do = act_space.sample()

        # Exploit : Otherwise, returns the best choice according to the DQN
        else :
            with torch.no_grad():
                action = self.net(torch.tensor(obs_new))
                what_to_do = action.numpy()


        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.time_step += 1

        return what_to_do



    def set(self, obs_old, act, rwd, obs_new, terminated):
        """
        Learn from one step
        """


        # Get y_pred
        out = self.net(obs_old)

        terminated = terminated.long()


        # Get y_max from target
        with torch.no_grad():
            exp = rwd + (1 - terminated) * CFG.gamma * self.target(obs_new)

        # Compute loss
        loss = torch.square(out - exp).mean()

        # Backward propagation
        # Gradient descent updates weight in the network to minimize loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def training(self,  render_every=100):

        total_rewards = []
        episodes = CFG.episodes
        max_steps = CFG.max_steps

        for episode in range(episodes):

            if episode %render_every == 0:
                env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')
            else :
                env = gym.make("BipedalWalker-v3",hardcore=False)


            # Start episode
            start_time=time.time()

            r = 0

            obs_old, _ = env.reset()

            env = RewardWrapper(env)

            for step in range(max_steps):

                # Get action from agent and give it to environment
                action = self.get(obs_old, env.action_space)
                obs_new, reward, terminated, truncated, _ = env.step(action)
                r += reward

                done = terminated or truncated

                # if reward < 0 :
                #     reward = 0
                # else :
                #     reward *= 100

                # Storing step into buffer
                BUF.set((obs_old, action, reward, obs_new, done))

                # Training self if buffer is full (no need to clear it because size)
                if self.time_step % CFG.buffer_size == 0 :
                    old_list, act_list, rwd_list, new_list, new_terminated = BUF.get()
                    self.set(old_list, act_list, rwd_list, new_list, new_terminated)


                # Updating target after sync_every steps
                if self.time_step % CFG.sync_every == 0:
                    self.target.load_state_dict(self.net.state_dict())
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
                print(self.exploration_rate)
        return max(total_rewards)


    def training_and_record(self):

        total_rewards = []
        episodes = CFG.episodes
        max_steps = CFG.max_steps
        step_starting_index =0

        for episode in range(episodes):


            env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='rgb_array_list')

            episode_index = 0


            # Start episode
            start_time=time.time()

            r = 0

            obs_old, _ = env.reset()

            env = RewardWrapper(env)

            for step in range(max_steps):

                # Get action from agent and give it to environment
                action = self.get(obs_old, env.action_space)
                obs_new, reward, terminated, truncated, _ = env.step(action)
                r += reward

                done = terminated or truncated

                # if reward < 0 :
                #     reward = 0
                # else :
                #     reward *= 100

                # Storing step into buffer
                BUF.set((obs_old, action, reward, obs_new, done))

                # Training self if buffer is full (no need to clear it because size)
                if self.time_step % CFG.buffer_size == 0 :
                    old_list, act_list, rwd_list, new_list, new_terminated = BUF.get()
                    self.set(old_list, act_list, rwd_list, new_list, new_terminated)


                # Updating target after sync_every steps
                if self.time_step % CFG.sync_every == 0:
                    self.target.load_state_dict(self.net.state_dict())
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
                print(self.exploration_rate)
            save_video(
                            env.render(),
                            f"{today}__{type(self).__name__}_train_{CFG.episodes}episodes__{CFG.max_steps}steps",
                            fps=env.metadata["render_fps"],
                            step_starting_index=step_starting_index,
                            episode_index=episode,
                            name_prefix = f""
                        )
            print('video saved ',episode,step_starting_index)

            step_starting_index =step + 1
            episode_index += 1

        return max(total_rewards)



    def evaluate(self, number_of_steps, recording):
        if recording==False:
            input("Press enter to see validation ")
            env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')
            obs_old, info = env.reset()
            terminated = False

            for step in range(number_of_steps):
                if terminated :
                    obs_old, info = env.reset()
                action = self.get(obs_old, env.action_space, evaluating=True)
                obs_new, reward, terminated, truncated, info = env.step(action)

            obs_old = obs_new

        else :
            input("Press enter to see validation ")
            env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='rgb_array_list')
            obs_old, info = env.reset()
            episode_index = 0
            step_starting_index =0
            terminated = False

            for step in range(number_of_steps):

                if terminated :
                    print(episode_index)
                    if episode_index%CFG.episodes_recording_freq==0:


                        save_video(
                            env.render(),
                            f"{today}__{type(self).__name__}____{number_of_steps}steps",
                            fps=env.metadata["render_fps"],
                            step_starting_index=step_starting_index,
                            episode_index=episode_index,
                            name_prefix = f""
                        )
                        print('video saved')

                    step_starting_index =step + 1
                    episode_index += 1

                    obs_old, info = env.reset()

                action = self.get(obs_old, env.action_space, evaluating=True)

                obs_new, reward, terminated, truncated, _ = env.step(action)

                obs_old = obs_new
            print('video saved')
