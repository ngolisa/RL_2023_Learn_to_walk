import gym
import torch

import pickle

# import data
# train model
# evaluate model



env = gym.make("BipedalWalker-v3",hardcore=True)

iterations = 1000
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
