


import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.stats import norm
import equinox as eqx
from optax import adam


# def get_random_policy_inverted_pendulum(num_episodes=10):
#     xy_pairs = []
#     env = gym.make("InvertedPendulum-v5")
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     for episode in range(num_episodes):
#         actions, states, next_states, cum_rewards, rewards = [],[],[],[],[]
#         ### make sure things are deterministic
#         env.action_space.seed(episode)
#         state, info = env.reset(seed=episode)
#         over = False

#         cum_reward = 0.
#         while not over:
#             action = env.action_space.sample()
#             next_state, reward, terminated, truncated, info = env.step(action)
#             cum_reward += reward
#             over = terminated or truncated
#             actions.append(action), states.append(state), next_states.append(next_state), rewards.append(reward)
#             cum_rewards.append(cum_reward)
#             state = next_state

#         a,s,sp1,r,cum_r = jnp.stack(actions), jnp.stack(states), jnp.stack(next_states), jnp.stack(rewards), jnp.stack(cum_rewards)
#         episode_len = len(a)
#         ts = jnp.arange(episode_len)
#         data = jnp.hstack((a,s,sp1,cum_r[:,None],ts[:,None]))

#         pairs = jnp.stack([data[:-1], data[1:]], axis=1) ### all tuples ((a_t, s_t, s_t+1, r_cum_t, t), (a_t+1, s_t+1, s_t+2, r_cum_t+1, t+1)) for nsde training
#         xy_pairs.append(pairs)
#     env.close()

#     raw_data = jnp.concatenate((xy_pairs), axis=0)
#     x = raw_data[:,0] #### (N,a+s+s'+r+t)
#     y = raw_data[:,1,-2:] ### cum rewards and time for next time step
#     data_cleaned = jnp.concatenate((x,y), axis=-1)
#     print(f"{data_cleaned.shape=}, {action_dim=}, {state_dim=}, columns ordered as: action + state + state' + rew_cum + t + rew_cum' + t' ")
#     return data_cleaned, raw_data, action_dim, state_dim


# data_clean, raw_data, state_dim, action_dim = get_random_policy_inverted_pendulum(num_episodes=10000)
# np.savez(f'random_policy_inverted_pendulum_{len(data_clean)}', data_clean=data_clean, raw_data=raw_data, state_dim=state_dim, action_dim=action_dim)






def get_random_policy_swimmer(num_episodes=10):
    xy_pairs = []
    env = gym.make("Swimmer-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    for episode in range(num_episodes):
        actions, states, next_states, cum_rewards, rewards = [],[],[],[],[]
        ### make sure things are deterministic
        env.action_space.seed(episode)
        state, info = env.reset(seed=episode)
        over = False

        cum_reward = 0.
        while not over:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward
            over = terminated or truncated
            actions.append(action), states.append(state), next_states.append(next_state), rewards.append(reward)
            cum_rewards.append(cum_reward)
            state = next_state

        a,s,sp1,r,cum_r = jnp.stack(actions), jnp.stack(states), jnp.stack(next_states), jnp.stack(rewards), jnp.stack(cum_rewards)
        episode_len = len(a)
        ts = jnp.arange(episode_len)
        data = jnp.hstack((a,s,sp1,cum_r[:,None],ts[:,None]))

        pairs = jnp.stack([data[:-1], data[1:]], axis=1) ### all tuples ((a_t, s_t, s_t+1, r_cum_t, t), (a_t+1, s_t+1, s_t+2, r_cum_t+1, t+1)) for nsde training
        xy_pairs.append(pairs)
    env.close()

    raw_data = jnp.concatenate((xy_pairs), axis=0)
    x = raw_data[:,0] #### (N,a+s+s'+r+t)
    y = raw_data[:,1,-2:] ### cum rewards and time for next time step
    data_cleaned = jnp.concatenate((x,y), axis=-1)
    print(f"{data_cleaned.shape=}, {action_dim=}, {state_dim=}, columns ordered as: action + state + state' + rew_cum + t + rew_cum' + t' ")
    return data_cleaned, raw_data, action_dim, state_dim


data_clean, raw_data, state_dim, action_dim = get_random_policy_swimmer(num_episodes=100)
np.savez(f'random_policy_swimmer_{len(data_clean)}', data_clean=data_clean, raw_data=raw_data, state_dim=state_dim, action_dim=action_dim)
