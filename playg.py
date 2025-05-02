from utils_for_d4rl_mujoco import (
    get_formatted_dataset_for_nsde_training,
    get_environment_infos_from_name,
    load_neorl_dataset,
)
from my_utils import GaussianKernel
import jax
import equinox as eqx
from jax import numpy as jnp, random as jr
from typing import List, Callable
import numpy as np
import diffrax


key = jr.PRNGKey(0)



task = 'Hopper-v3-Medium-1000-neorl'
dataset, env = load_neorl_dataset(task, return_env=True)
full_data = get_formatted_dataset_for_nsde_training(task)

print(full_data)
obs_shape = env.observation_space.shape
action_dim = np.prod(env.action_space.shape)

obs_pos_dim = 5
obs_vel_dim = obs_pos_dim ### otherwise dimensions don't match up. 
obs_full_dim = obs_pos_dim + obs_vel_dim

### Group neural networks according to the main loss terms 


### DIFFUSION TERM STUFF ( sig(s,a) + h(eta(s,a)) ) * dW ---> SHOULD OUTPUT state dim, whcih plugged into diagonal of big Sigma

sig_depth = 2
sig_latent_dim = 256
sig_act = jax.nn.tanh
sig_input_dim = obs_full_dim + action_dim
sig_out_dim = obs_pos_dim


### distance aware from dataset term
eta_depth = 2
eta_input_dim = obs_full_dim + action_dim
eta_latent_dim = 64
eta_act = jax.nn.swish
eta_out_dim = 1

### h(eta) -- monotonic and bounded as a function of eta --> W_max * sigmoid(W eta + b), W > 1
h_input_dim = 1
h_out_dim = obs_full_dim
W_max = 1.  ### THIS IS NOT REPORTED


### ALL MLPs have: depth, latent_dim, input_dim, activation, output_dim, last layer activation?

### DRIFT TERM STUFF f(s,a) dt--> [s_vel, G(s_vel) @ a + H(s) @ s_vel] + f_res --> organize state into pos,vel components at the outset
G_input_dim = obs_vel_dim
G_depth = 3
G_latent_dim = 256
G_act = jax.nn.swish
G_out_dim = obs_vel_dim*action_dim ### matrix
H_input_dim = obs_full_dim
H_depth, H_latent_dim, H_act = G_depth, G_latent_dim, G_act
H_out_dim = obs_vel_dim*obs_vel_dim ### matrix

f_res_input_dim = obs_full_dim + action_dim
f_res_depth, f_res_act, f_res_latent_dim, f_res_out_dim = G_depth, G_latent_dim, G_act, obs_full_dim


f_rew_depth = 3
f_rew_input_dim = obs_full_dim + action_dim
f_rew_act = jax.nn.swish
f_rew_latent_dim = 64
f_rew_out_dim=1


### strong convexity constraint nn
L_mu_input_dim = obs_full_dim + action_dim
L_mu_out_dim = 1
L_mu_depth = 2
L_mu_latent_dim = 32
L_mu_act = jax.nn.swish


### how to sample for L_sc term 



### concatentate 0 to the diffusion term output because r is deterministic 

class DriftTerm(eqx.Module):
    G: eqx.Module
    H: eqx.Module
    f_rew: eqx.Module
    f_res: eqx.Module


    def __init__(self, *, key):
        keys = jr.split(key, 4)
        self.G = lambda s_vel: eqx.nn.MLP(in_size=G_input_dim,
                            out_size=G_out_dim,
                            width=G_latent_dim,
                            depth=G_depth,
                            activation=G_act,
                            final_activation=None,
                            key=keys[0]
                            )(s_vel).reshape(obs_vel_dim, action_dim)
        
        self.H = lambda s: eqx.nn.MLP(in_size=H_input_dim,
                            out_size=H_out_dim,
                            width=H_latent_dim,
                            depth=H_depth,
                            activation=H_act,
                            final_activation=None,
                            key=keys[1]
                            )(s).reshape(obs_vel_dim, obs_vel_dim)
        
        self.f_res =  eqx.nn.MLP(in_size=f_res_input_dim,
                            out_size=f_res_out_dim,
                            width=f_res_latent_dim,
                            depth=f_res_depth,
                            activation=f_res_act,
                            final_activation=None,
                            key=keys[2]
                            )
        
        self.f_rew =  eqx.nn.MLP(in_size=f_rew_input_dim,
                            out_size=f_rew_out_dim,
                            width=f_rew_latent_dim,
                            depth=f_rew_depth,
                            activation=f_rew_act,
                            final_activation=None,
                            key=keys[2]
                            )

    #  f(s,a) dt--> [s_vel, G(s_vel) @ a + H(s) @ s_vel] + f_res
    def __call__(self, t, y, args):
        action = args ### action_dim,
        s_pos = y[:obs_pos_dim]
        s_vel = y[obs_pos_dim:obs_pos_dim+obs_vel_dim]
        s = y[:obs_pos_dim+obs_vel_dim]
        
        f_vel = self.G(s_vel) @ action + self.H(s) @ s_vel
        f_pos = s_vel

        f_known = jnp.vstack((f_pos, f_vel))

        s_action = jnp.vstack((s, action))
        f_res = self.f_res(s_action)
        
        f = f_known + f_res
        
        f_rew = self.f_rew(s_action)

        return jnp.vstack((f,f_rew))


#  ( sig(s,a) + h(eta(s,a)) ) * dW

class DiffusionTerm(eqx.Module):
    eta: eqx.Module
    h: Callable
    sigma: eqx.Module

    def __init__(self, *, key):

        keys = jr.split(key, 3)

        self.eta = eqx.nn.MLP(in_size=eta_input_dim,
                            out_size=eta_out_dim,
                            width=eta_latent_dim,
                            depth=eta_depth,
                            activation=eta_act,
                            final_activation=None,
                            key=keys[0]
                            )
        
        ###
        W = jr.uniform(keys[1], (h_out_dim,))
        b = jr.zeros((h_out_dim,))
        ### constrain W to be greater than 1
        self.h = lambda eta: W_max * (jax.nn.softplus(W)+1) * eta + b

        self.sigma = eqx.nn.MLP(in_size=sig_input_dim,
                            out_size=sig_out_dim,
                            width=sig_latent_dim,
                            depth=sig_depth,
                            activation=sig_act,
                            final_activation=None,
                            key=keys[2]
                            )

    def __call__(self, t, y, args):
        action = args
        s = y[:-1] ### exclude reward
        s_action = jnp.vstack((s, action))
        Sigma_diagonal = self.sigma(s_action) + self.h(self.eta(s_action))
        Sigma_diagonal = jnp.pad(Sigma_diagonal, 1) ### pad for the reward
        Sigma = jnp.diag(Sigma_diagonal)
        return Sigma ### obs_full_dim+1, obs_full_dim+1
    


class NeuralSDE(eqx.Module):
    drift: eqx.Module
    diffusion: eqx.Module
    L_mu: eqx.Module

    def __init__(self, *, key):
        keys = jr.split(key,3)
        self.diffusion = DiffusionTerm(key=keys[0])
        self.drift = DriftTerm(key=keys[1])
        self.L_mu = eqx.nn.MLP(in_size=L_mu_input_dim,
                            out_size=L_mu_out_dim,
                            width=L_mu_latent_dim,
                            depth=L_mu_depth,
                            activation=L_mu_act,
                            final_activation=jax.nn.softplus,
                            key=keys[2]
                            )

    def __call__(self, y0, t0, t1,*, key):
        dt0 = (t1-t0) /2
        bm_key,_ = jr.split(key)

        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0 / 2, shape=(obs_full_dim+1,), key=bm_key
        )  
        diffusion_term = diffrax.ControlTerm(self.diffusion, control)
        drift_term = diffrax.ODETerm(self.drift)
        vf_terms = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.ReversibleHeun()
        
        saveat = diffrax.SaveAt([t0,t1])
        sol = diffrax.diffeqsolve(vf_terms, solver, t0, t1, dt0, y0, saveat=saveat)
        return sol.ys[-1]
    

model = NeuralSDE(key=key)

### loss terms, one example
## lambda is 10^-4
def L_grad(state, action):
    s_action = jnp.vstack((state, action))
    eta_grad = jax.grad(model.diffusion.eta)(s_action)
    eta_squared_norm = eta_grad @ eta_grad
    eta_squared = model.diffusion.eta(s_action) ** 2
    return eta_squared_norm + eta_squared

### lambda is not reported, set it to 1
def L_sc(state, action, next_state_true, next_action):
    s_action_next = jnp.vstack((next_state_true, next_action))
    s_action = jnp.vstack((state, action))

    a = model.diffusion.eta(s_action_next)
    b = model.diffusion.eta(s_action)
    c = jax.grad(model.diffusion.eta)(s_action) @ (s_action - s_action_next)
    d = model.L_mu(s_action) * ((s_action - s_action_next) @ (s_action - s_action_next))
    convexity_constraint = a - b - c - d

# given data pair (s,a), sample in radius of 0.1, make a KD tree of the data....

### lambda is 1
def L_mu(state, action):
    s_action = jnp.vstack((state, action))
    return model.L_mu(s_action)

### 20 SAMPLES
### ball of radius one, strong ocnvexity coef of 1



### predicitng cumultiave reward also 