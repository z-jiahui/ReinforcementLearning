import gym
import numpy as np

def policy(q_values, action_dim, state, epsilon=0.1):
  '''
  epsilon-greedy policy
  '''
  if np.random.random() < epsilon:
    return np.random.choice(action_dim)
  else:
    return np.argmax(q_values[state])
  
def train(env, learning_rate=0.5, epsilon=0.1,
          gamma=0.9, iterations=10000, on_policy=True):
  q_values = np.zeros((env.observation_space.n, env.action_space.n))
  state_dim = q_values.shape[0]
  action_dim = q_values.shape[1]
  for _ in range(iterations):
    state = env.reset()
    done = False
    # Q-learning
    if not on_policy:
      while not done:
        action = policy(q_values, action_dim, state)
        next_state, reward, done, info = env.step(action)
        # update Qtable
        td_target = reward + gamma * np.max(q_values[next_state])
        td_error = td_target - q_values[state][action]
        q_values[state][action] += learning_rate * td_error
        # update A, S
        state = next_state
    # sarsa
    else:
      action = policy(q_values, action_dim, state)
      while not done:
        # take action, observe R',S';
        next_state, reward, done, info = env.step(action)
        # choose A' from S'
        next_action = policy(q_values, action_dim, next_state)
        # update Qtable
        td_target = reward + gamma * q_values[next_state][next_action]
        td_error = td_target - q_values[state][action]
        q_values[state][action] += learning_rate * td_error
        # update A, S
        action, state = next_action, next_state
  return q_values

def evaluate(env, episode, q_values, action_dim, epsilon=0.1):
  reward = []
  for _ in range(episode):
    state = env.reset();
    action = policy(q_values, action_dim, state, epsilon)
    done = False
    ep_reward = 0
    while not done:
      env.render()
      state, r, done, info = env.step(action)
      action = policy(q_values, action_dim, state, epsilon)
      ep_reward += r
    reward.append(ep_reward)
  return np.mean(reward)

if __name__ == "__main__":
  env = gym.make('FrozenLake-v0')
  env.seed(0)
  np.random.seed(0)
  ql_values = train(env, on_policy=False)
  ql_eval = evaluate(env, 1, ql_values, 4)
  sr_values = train(env)
  sr_eval = evaluate(env, 1, sr_values, 4)
  print("1")
