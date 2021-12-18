import sys
sys.path.append("/root/jiang21c-supp/MRPO-source-code/MRPOCode_ICML")
import gym, roboschool
from examples.MRPO import base


env_id = 'SunblazeWalker2dRandomNormal-v0'
env = base.make_env(env_id)
obs = env.reset()
h = 0
while True:
    print(obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print("{} timesteps, rew {}".format(h+1, reward))
        # break
        obs = env.reset()
        continue
    h += 1
print('finish')


# env_id = 'RoboschoolWalker2d-v1'
# env = gym.make(env_id)
# obs = env.reset()
# h = 0
# while True:
#     print(obs)
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     if done:
#         print("Episode finished after {} timesteps".format(h+1))
#         break
#     h += 1
# print('finish')


# env_id = 'Walker2d-v2'
# env = gym.make(env_id)
# obs = env.reset()
# h = 0
# while True:
#     print(obs)
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     if done:
#         print("Episode finished after {} timesteps".format(h+1))
#         break
#     h += 1
# print('finish')