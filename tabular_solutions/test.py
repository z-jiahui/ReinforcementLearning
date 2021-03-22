import gym
env = gym.make('CartPole-v0')
obsdim = env.observation_space.shape
reward = []
for i_episode in range(10):
    observation = env.reset()
    ep_reward = 0
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, r, done, info = env.step(action)
        ep_reward += r
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            reward.append(ep_reward)
            ep_reward = 0
            break
env.close()