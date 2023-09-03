import gym

from stable_baselines3.sac.policies import MlpPolicy,MultiInputPolicy
# from stable_baselines3.common import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
import os
from stable_baselines3.common.vec_env import DummyVecEnv



# import tensorflow as tf
# import tensorflow_addons as tfa

env = gym.make("BipedalWalker-v3", hardcore=True)
env = DummyVecEnv([lambda: env])

env = make_vec_env("BipedalWalker-v3",n_envs=2**6)


model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=5_000_000)
# model.save(os.path.join(os.path.dirname(__file__),'data/',f"a2c"))

# del model # remove to demonstrate saving and loading

model = A2C.load(os.path.join(os.path.dirname(__file__),'data/',f"a2c"))
# model = A2C.load('a2c') #modele entrainé sur le hardcore = False

# env = gym.make("BipedalWalker-v3")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

input('press enter to see the eval')
eval_env = Monitor(gym.make("BipedalWalker-v3",hardcore = True, render_mode='human'))
evaluate_policy(model, eval_env, n_eval_episodes=10)
