import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.stats import norm
import equinox as eqx
from optax import adam


state_dim, action_dim = 4, 1 ### for pendulum 
state_pos_dim = state_dim // 2 ### equal number of position dims in state vector as in velocity dims
state_vel_dim = state_pos_dim

### Group neural networks according to the main loss terms 


### DIFFUSION TERM STUFF ( sig(s,a) + h(eta(s,a)) ) * dW ---> SHOULD OUTPUT state dim, whcih plugged into diagonal of big Sigma
sig_depth = 2
sig_latent_dim = 256
sig_act = jax.nn.tanh
sig_input_dim = state_dim + action_dim
sig_out_dim = state_dim


### distance aware from dataset term
eta_depth = 2
eta_input_dim = state_dim + action_dim
eta_latent_dim = 64
eta_act = jax.nn.swish
eta_out_dim = 1

### h(eta) -- monotonic and bounded as a function of eta --> W_max * sigmoid(W eta + b), W > 1
h_input_dim = 1
h_out_dim = state_dim
W_max = 1.  ### THIS IS NOT REPORTED


### DRIFT TERM STUFF f(s,a) dt--> [s_vel, G(s_vel) @ a + H(s) @ s_vel] + f_res --> organize state into pos,vel components at the outset
G_input_dim = state_vel_dim
G_depth = 3
G_latent_dim = 256
G_act = jax.nn.swish
G_out_dim = state_vel_dim*action_dim ### matrix
H_input_dim = state_dim
H_depth, H_latent_dim, H_act = G_depth, G_latent_dim, G_act
H_out_dim = state_vel_dim*state_vel_dim ### matrix

f_res_input_dim = state_dim + action_dim
f_res_depth, f_res_act, f_res_latent_dim, f_res_out_dim = G_depth, G_act, G_latent_dim, state_dim

f_rew_depth = 3
f_rew_input_dim = state_dim + action_dim
f_rew_act = jax.nn.swish
f_rew_latent_dim = 64
f_rew_out_dim=1


### strong convexity constraint nn
L_mu_input_dim = state_dim + action_dim
L_mu_out_dim = 1
L_mu_depth = 2
L_mu_latent_dim = 32
L_mu_act = jax.nn.swish