import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.stats.norm import logpdf
import equinox as eqx
from optax import adam

seed = 42
np.random.seed(seed)
# env = gym.make("Hopper-v5", exclude_current_positions_from_observation=False)
env = gym.make("InvertedPendulum-v5")
state_dim    = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = 3.
key = jr.PRNGKey(seed)


key, k1 = jr.split(key)

actor = eqx.nn.MLP(
    in_size=state_dim,
    out_size=action_dim*2,
    width_size=64,
    depth=3,
    activation=jax.nn.relu,
    key=key,
)

critic = eqx.nn.MLP(
    in_size=state_dim,
    out_size=1,
    width_size=64,
    depth=3,
    activation=jax.nn.relu,
    key=k1,
)

gamma = 0.99
opt_actor = adam(3e-4)
opt_state_actor = opt_actor.init(eqx.filter(actor, eqx.is_array))

opt_critic = adam(3e-4)
opt_state_critic = opt_critic.init(eqx.filter(critic, eqx.is_array))


@eqx.filter_jit
def train_step(actor, opt_state_actor, critic, opt_state_critic, batch):
    states, next_states, actions, action_dists, rewards = batch
    def loss_critic(critic):
        values = eqx.filter_vmap(critic)(states).squeeze()
        next_values = eqx.filter_vmap(critic)(next_states).squeeze()
        td_target = rewards + gamma * next_values
        advantage = td_target - values
        se_loss = (advantage**2).sum(axis=-1)
        mse_loss = se_loss.mean()
        return mse_loss, advantage

    ### update critic model 
    (loss_critic, advantage), grads = eqx.filter_value_and_grad(loss_critic, has_aux=True)(critic)
    updates, opt_state_critic = opt_critic.update(
        grads, opt_state_critic, eqx.filter(critic, eqx.is_array)
    )
    critic = eqx.apply_updates(critic, updates)
    
    def loss_actor(actor):
        logp = logpdf(actions, loc=action_dists[:,0], scale=action_dists[:,1]).sum(-1)
        return -(logp * advantage).mean()

    ### update actor model
    loss_actor, grads = eqx.filter_value_and_grad(loss_actor)(actor)
    updates, opt_state_actor = opt_actor.update(
        grads, opt_state_actor, eqx.filter(actor, eqx.is_array)
    )
    actor = eqx.apply_updates(actor, updates)

    return actor, loss_actor, opt_state_actor, critic, loss_critic, opt_state_critic


def collect_episode(actor, key):
    states, next_states, action_samples, action_dists, rewards = [], [], [], [], []
    state, _ = env.reset()
    done = False
    while not done:

        ### sample action
        action_dist = actor(state)
        mu,std = jnp.split(action_dist, 2, axis=-1)
        mu = jax.nn.tanh(mu) * action_scale
        std = jax.nn.softplus(std) + 1e-6

        key, sub = jr.split(key)
        action_sample = mu + std * jr.normal(sub, (action_dim,))

        ### take step
        next_state, reward, term, trunc, _ = env.step(action_sample)

        ### record
        states.append(state), next_states.append(next_state), action_samples.append(action_sample), rewards.append(reward)
        action_dists.append(jnp.vstack((mu, std)))

        state = next_state
        done = term or trunc

    return jnp.stack(states), jnp.stack(next_states), jnp.stack(action_samples).squeeze(), jnp.stack(action_dists).squeeze(), jnp.array(rewards)


for ep in range(1000):

    ### run episode
    key, episode_key = jr.split(key)
    batch = collect_episode(actor, episode_key)

    print(f"{ep=} length:", len(batch[0]), end=",")
    
    ### update models
    for _ in range(1):
        actor, loss_actor, opt_state_actor, critic, loss_critic, opt_state_critic = train_step(actor, opt_state_actor, critic, opt_state_critic, batch)
    print(f'{loss_actor.item()=:.3f}, {loss_critic.item()=:.3f}')

env.close()