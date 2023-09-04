import gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_stuck = 0


    def step(self, action):

        action = np.around(3/2*action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if obs[2]<=.2:
            self.time_stuck += 1
            if self.time_stuck >=30:
                reward -= .01
        elif obs[2] >= .2:
            self.time_stuck = 0


        # if obs[8] + obs[13] == 1:
        #     reward += 1

        # if np.abs(obs[0]) <= .5:
        #     reward += 10
        # else:
        #     reward-=10

        # if (obs[6]<-1.57 and obs[13]==1) or (obs[11]<-1.57 and obs[8]==1):
        #     reward +=.5

        # if (obs[5]*obs[7]<0) or (obs[10]*obs[12]<0):
        #     reward +=5



        # if obs[2]>.3:
        #     reward+=100

        # if max(obs[7]/obs[12], obs[7]/obs[12])  <= 5:
        #     reward -=1
        # # else :
        # #     reward -= 1
        return obs, reward, terminated, truncated, info
