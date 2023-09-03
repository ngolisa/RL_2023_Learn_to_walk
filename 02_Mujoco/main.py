import gym

from stable_baselines3.sac.policies import MlpPolicy,MultiInputPolicy
# from stable_baselines3.common import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
import os
from stable_baselines3.common.vec_env import DummyVecEnv



# import tensorflow as tf
# import tensorflow_addons as tfa

env = gym.make("Humanoid-v4")
env = DummyVecEnv([lambda: env])

env = make_vec_env("Humanoid-v4",n_envs=2**7)


model = A2C('MlpPolicy', env, verbose=1)
# model.load(os.path.join(os.path.dirname(__file__),'data_mujoco/',f"a2c_muj_4"))

model.learn(total_timesteps=100_000_000)
model.save(os.path.join(os.path.dirname(__file__),'data_mujoco/',f"a2c_muj_100_000_000"))

#learn = (,3_000_000) + (2, 3_000_000) + (3, 3_000_000) + (4, 1_000_000) + (5, 3_000_000)

# del model # remove to demonstrate saving and loading

# model.load(os.path.join(os.path.dirname(__file__),'data_mujoco/',f"a2c_muj"))
# model = A2C.load('a2c') #modele entrainé sur le hardcore = False

# model.learn(total_timesteps=1_000_000)
# model.save(os.path.join(os.path.dirname(__file__),'data_mujoco/',f"a2c"))



# env = gym.make("Humanoid-v4")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

input('press enter to see the eval')
eval_env = Monitor(gym.make("Humanoid-v4", render_mode='human'))
rew = evaluate_policy(model, eval_env, n_eval_episodes=1000)

print('rew', rew)
