import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.stats.norm import logpdf
import equinox as eqx
from optax import adamw

seed = 42
np.random.seed(seed)
# env = gym.make("Hopper-v5", exclude_current_positions_from_observation=False)
env = gym.make("InvertedPendulum-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = 3.
key = jr.PRNGKey(seed)


key, k1 = jr.split(key)

actor = eqx.nn.MLP(
    in_size=state_dim,
    out_size=action_dim,
    width_size=64,
    depth=3,
    activation=jax.nn.relu,
    final_activation=lambda x: jax.nn.tanh(x)*action_scale,
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
opt_actor = adamw(1e-3)
opt_state_actor = opt_actor.init(eqx.filter(actor, eqx.is_array))

opt_critic = adamw(1e-8)
opt_state_critic = opt_critic.init(eqx.filter(critic, eqx.is_array))

def sample_action(actor, state, key):
    action_mu = actor(state)
    std = jnp.ones_like(action_mu) * 0.5
    key, sub = jr.split(key)
    action_sample = action_mu + std * jr.normal(sub, (action_dim,))
    return action_sample, (action_mu, std)


episodes = 1000
for episode in range(episodes):

    state, _ = env.reset()
    done = False

    episode_length = 0
    while not done:

        key,_ = jr.split(key)

        ### take step
        action_sample,(_,_) = sample_action(actor, state, key)
        next_state, reward, term, trunc, _ = env.step(action_sample)

        ####################################################################################################
        ######## UPDATE CRITIC ##############################################################################
        # @eqx.filter_jit
        def loss_critic(critic):
            value = critic(state).squeeze()
            next_value = critic(next_state).squeeze()
            td_target = reward + gamma * next_value
            advantage = td_target - value
            mse_loss = (advantage**2)
            return mse_loss, advantage
        (loss_critic, advantage), grads = eqx.filter_value_and_grad(loss_critic, has_aux=True)(critic)
        updates, opt_state_critic = opt_critic.update(
            grads, opt_state_critic, eqx.filter(critic, eqx.is_array)
        )
        critic = eqx.apply_updates(critic, updates)
        
        ####################################################################################################
        ######## UPDATE ACTOR ##############################################################################
        # @eqx.filter_jit
        def loss_actor(actor):
            action_sample, (mu, std) = sample_action(actor, state, key) ### same sample as before, but in computational graph
            logp = logpdf(action_sample, mu, std).sum()
            return -(logp * advantage)
        loss_actor, grads = eqx.filter_value_and_grad(loss_actor)(actor)
        updates, opt_state_actor = opt_actor.update(
            grads, opt_state_actor, eqx.filter(actor, eqx.is_array)
        )
        actor = eqx.apply_updates(actor, updates)

        print(f'{loss_actor.item()=:.3f}, {loss_critic.item()=:.3f}')
        episode_length+=1
        state = next_state
        done = term or trunc

    print(f'{episode_length=}')