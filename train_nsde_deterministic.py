import jax
import equinox as eqx
from jax import numpy as jnp, random as jr
from typing import List, Callable
import numpy as np
import diffrax
from optax import adam, adamw

### not the best code practice, but sets variables including data
from run_nsde_setup_pend import *
# from run_nsde_setup_swimmer import *


import wandb


wandb.init()

key = jr.PRNGKey(0)

class DriftTerm(eqx.Module):
    G: eqx.Module
    H: eqx.Module
    f_rew: eqx.Module
    f_res: eqx.Module


    def __init__(self, *, key):
        keys = jr.split(key, 4)
        self.G = lambda s_vel: eqx.nn.MLP(in_size=G_input_dim,
                            out_size=G_out_dim,
                            width_size=G_latent_dim,
                            depth=G_depth,
                            activation=G_act,
                            key=keys[0]
                            )(s_vel).reshape(state_vel_dim, action_dim)
        
        self.H = lambda s: eqx.nn.MLP(in_size=H_input_dim,
                            out_size=H_out_dim,
                            width_size=H_latent_dim,
                            depth=H_depth,
                            activation=H_act,
                            key=keys[1]
                            )(s).reshape(state_vel_dim, state_vel_dim)
        
        self.f_res =  eqx.nn.MLP(in_size=f_res_input_dim,
                            out_size=f_res_out_dim,
                            width_size=f_res_latent_dim,
                            depth=f_res_depth,
                            activation=f_res_act,
                            key=keys[2]
                            )
        
        self.f_rew =  eqx.nn.MLP(in_size=f_rew_input_dim,
                            out_size=f_rew_out_dim,
                            width_size=f_rew_latent_dim,
                            depth=f_rew_depth,
                            activation=f_rew_act,
                            key=keys[2]
                            )

    #  f(s,a) dt--> [s_vel, G(s_vel) @ a + H(s) @ s_vel] + f_res
    def __call__(self, t, y, args):
        action = args ### action_dim,
        s_pos = y[:state_pos_dim]
        s_vel = y[state_pos_dim:state_pos_dim+state_vel_dim]
        s = y[:state_pos_dim+state_vel_dim]
        
        f_vel = self.G(s_vel) @ action + self.H(s) @ s_vel
        f_pos = s_vel

        f_known = jnp.concatenate((f_pos, f_vel))

        s_action = jnp.concatenate((s, action))
        f_res = self.f_res(s_action)
        
        f = f_known + f_res
        
        f_rew = self.f_rew(s_action)

        out = jnp.concatenate((f,f_rew))
        return out


#  ( sig(s,a) + h(eta(s,a)) ) * dW

class DiffusionTerm(eqx.Module):
    eta: eqx.Module
    h_W: jax.Array
    h_b: jax.Array
    sigma: eqx.Module

    def __init__(self, *, key):

        keys = jr.split(key, 3)

        self.eta = eqx.nn.MLP(in_size=eta_input_dim,
                            out_size=eta_out_dim,
                            width_size=eta_latent_dim,
                            depth=eta_depth,
                            activation=eta_act,
                            key=keys[0]
                            )
        
        self.h_W = jr.uniform(keys[1], (h_out_dim,))
        self.h_b = jnp.zeros((h_out_dim,))

        self.sigma = eqx.nn.MLP(in_size=sig_input_dim,
                            out_size=sig_out_dim,
                            width_size=sig_latent_dim,
                            depth=sig_depth,
                            activation=sig_act,
                            key=keys[2]
                            )

    def __call__(self, t, y, args):
        action = args
        h = lambda eta: W_max * (jax.nn.softplus(self.h_W)+1) * eta + self.h_b
        s = y[:-1] ### exclude reward, shape = 4? 
        s_action = jnp.concatenate((s, action))
        Sigma_diagonal = self.sigma(s_action) + h(self.eta(s_action))
        Sigma_diagonal = jnp.pad(Sigma_diagonal, (0, 1)) ### pad for the reward
        Sigma = jnp.diag(Sigma_diagonal)
        return Sigma ### state_dim+1, state_dim+1
    


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
                            width_size=L_mu_latent_dim,
                            depth=L_mu_depth,
                            activation=L_mu_act,
                            final_activation=jax.nn.softplus,
                            key=keys[2]
                            )

    def __call__(self, y0, t0, t1, args, *, key):

        bm_key,_ = jr.split(key)
        t0 = t0[0]
        t1 = t1[0]
        ### literally just looking to solve for t1 given t0. In the paper they set the step size to be exactly t1-t0, but maybe thats
        ### because they didn't use a good sde solver library like this one, where we can set it to be anything easily
        control = diffrax.VirtualBrownianTree(t0=t0, t1=t1, tol=0.1, shape=(state_dim+1,), key=bm_key)
        diffusion_term = diffrax.ControlTerm(self.diffusion, control)

        drift_term = diffrax.ODETerm(self.drift)

        vf_terms = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.ReversibleHeun()
        saveat = diffrax.SaveAt(t0=True, t1=True)

        sol = diffrax.diffeqsolve(vf_terms, solver, t0=t0, t1=t1, dt0=0.1, y0=y0, args=args, saveat=saveat)
        return sol.ys[-1]
    

model = NeuralSDE(key=key)

### loss terms, written for one s_t, a_t pair, save for L_mu, which is written for a batch

## lambda is 10^-4
def L_grad(model, state, action):
    s_action = jnp.concatenate((state, action))
    eta_grad = jax.jacfwd(model.diffusion.eta)(s_action).squeeze()
    eta_squared_norm = eta_grad @ eta_grad
    eta_squared = model.diffusion.eta(s_action) ** 2
    return eta_squared_norm + eta_squared

### EQUATION 3, page 6. Note: lambda / coefficient for this loss term is not reported, set it to one
def L_sc(model, state, action, key, radius=0.1, num_samples=20):

    ### here, we get the s/a pair, take (20,2) samples from the rest of the dataset a normal distribution w/ std r centered

    s_action = jnp.concatenate((state, action))

    key,_ = jr.split(key)
    s_action_s_action_next_samples = jr.normal(key, (num_samples, 2, len(s_action))) * radius + s_action

    def convexity_constraint(s_action_s_action_next_samples):
        s_action = s_action_s_action_next_samples[0]
        s_action_next = s_action_s_action_next_samples[1]

        a = model.diffusion.eta(s_action_next)
        b = model.diffusion.eta(s_action)
        c = jax.jacfwd(model.diffusion.eta)(s_action).squeeze() @ (s_action - s_action_next)
        d = model.L_mu(s_action) * ((s_action - s_action_next) @ (s_action - s_action_next))
        return a - b - c - d
        
    convexity_constraints = jax.vmap(convexity_constraint)(s_action_s_action_next_samples)
    out = jnp.where(convexity_constraints >= 0, 0, convexity_constraints**2)
    return out.sum()


### EQUATION 3, page 6. lambda = 1
def L_mu(model, state_batch, action_batch):
    s_action_batch = jnp.concatenate((state_batch, action_batch), axis=-1) ### batch_size, s_dim + action_dim
    mus = eqx.filter_vmap(model.L_mu)(s_action_batch)
    return (1/mus).sum()

#### assume one sample as the paper mentions in page 6, equation 6 
def L_data(model, state, state_next, action, cum_r, cum_r_next, t0, t1, key):
    #### model digests state and current cumulative reward
    state_r = jnp.concatenate((state,cum_r))

    key,_= jr.split(key)
    state_r_next_pred_sample = model(state_r, t0, t1, action, key=key) ### do the solve

    ### separate out state pred and cum reward pred
    state_next_pred_sample = state_r_next_pred_sample[:-1]
    cum_r_next_pred = state_r_next_pred_sample[-1]

    sigma = model.diffusion(t0, state_r, action)
    sigma = sigma[:-1,:-1] ### again, strip cum reward portion
    sigma_log_det = jax.numpy.linalg.slogdet(sigma)[1]

    diff = state_next - state_next_pred_sample
    # sigma_norm_squared_diff = diff @  (1/sigma) @ diff
    # loss = sigma_norm_squared_diff + sigma_log_det
    loss = jnp.linalg.norm(diff)
    loss += jnp.abs(cum_r_next - cum_r_next_pred)
    return loss


def cyclical_cosine_annealing(
    total_steps,
    init_value=1e-4,
    warmup_frac=0.3,
    peak_value=3e-4,
    end_value=1e-4,
    num_cycles=6,
    gamma=0.9,
):
    decay_steps = total_steps / num_cycles
    schedules = []
    boundaries = []
    boundary = 0
    for cycle in range(num_cycles):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            warmup_steps=decay_steps * warmup_frac,
            peak_value=peak_value,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=2,
        )
        boundary = decay_steps + boundary
        boundaries.append(boundary)
        init_value = end_value
        peak_value = peak_value * gamma
        schedules.append(schedule)
        
    return optax.join_schedules(schedules=schedules, boundaries=boundaries)

### data is ntrain, 2, action + state + state' + rew_cum + rew"
opt = adamw(1e-4)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

batch_split = jnp.cumsum(jnp.array([action_dim,state_dim,state_dim,1,1,1,1,]))[:-1].tolist()

lamb_data, lamb_sc, lamb_grad, lamb_mu = 1.,1.,1e-4,0.005

@eqx.filter_jit
def train_step(model, opt_state, batch, key):
    action_b, state_b, state_next_b, cum_r_b, t0_b, cum_r_next_b, t1_b = jnp.split(batch, batch_split, axis=-1)

    def loss(model):
        
        ### 1, this includes the reward loss
        loss_data = eqx.filter_vmap(lambda s, s_next, a, cum_r, cum_r_next, t0, t1: \
                                                L_data(model, s, s_next, a, cum_r, cum_r_next, t0, t1, key),
                                   in_axes=(0,)*7)

        loss_data = lamb_data * loss_data(state_b, state_next_b, action_b, cum_r_b, cum_r_next_b, t0_b, t1_b).mean()

        ### 2
        # loss_sc = lamb_sc * eqx.filter_vmap(lambda s, a: L_sc(model,s,a,key), in_axes=(0,0))(state_b, action_b).mean()
        
        ### 3
        # loss_grad = lamb_grad * eqx.filter_vmap(lambda s,a: L_grad(model, s,a), in_axes=(0,0))(state_b, action_b).mean()

        ### 4
        # loss_mu = lamb_mu * L_mu(model, state_b, action_b)
        
        # total_loss = loss_data+loss_sc+loss_grad+loss_mu
        total_loss = loss_data

        return  total_loss, (loss_data,)
    
    (loss, individal_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, individal_losses


@eqx.filter_jit
def eval_step(model, batch, key):
    action_b, state_b, state_next_b, cum_r_b, t0_b, cum_r_next_b, t1_b = jnp.split(batch, batch_split, axis=-1)
    def loss(model):

        def one(state, state_next, action, cum_r, cum_r_next, t0, t1, key):
            ### just compute squared differences for s_preds and r_preds for sanity's sake 
            state_r = jnp.concatenate((state,cum_r))
            key,_= jr.split(key)
            state_r_next_pred_sample = model(state_r, t0, t1, action, key=key) ### do the solve

            ### separate out state pred and cum reward pred
            state_next_pred_sample = state_r_next_pred_sample[:-1]
            cum_r_next_pred = state_r_next_pred_sample[-1]

            state_err_rel_l2 = jnp.linalg.norm(state_next - state_next_pred_sample) / jnp.linalg.norm(state_next)
            reward_err = jnp.abs(cum_r_next - cum_r_next_pred)
            return state_err_rel_l2, reward_err

        return jax.vmap(one, in_axes=(0,0,0,0,0,0,0,None))(state_b, state_next_b, action_b, cum_r_b, cum_r_next_b, t0_b, t1_b, key)


    state_err_rel_l2, reward_err = loss(model)
    return state_err_rel_l2.mean(), reward_err.mean()

### normalizing????? 


epochs = 10000
ntrain, ntest = 400, 100
data = np.load('random_policy_inverted_pendulum_563.npz')
# data = np.load('random_policy_swimmer_99900.npz')
data = data['data_clean']
data = jr.permutation(key, data)
train, test = data[:ntrain], data[ntrain:ntrain+ntest]
print(train.shape, test.shape)
batch_size = 400
num_train_batch = ntrain // batch_size


wandb.init(project='nsde_rl')

@jax.jit
def get_train_batch(
    i,
    key,
):  
    tr = jr.permutation(key, train)
    train_b = jax.lax.dynamic_slice_in_dim(tr,i * batch_size, batch_size)
    return train_b



for epoch in range(epochs):
    key,epoch_key = jr.split(key)

    for batch_i in range(num_train_batch):
        batch = get_train_batch(batch_i, epoch_key)
        model, opt_state, train_loss, individal_losses = train_step(model, opt_state, batch, epoch_key)

    state_err, rew_err = eval_step(model, train, key)
    test_state_err, test_rew_err = eval_step(model, test, key)
    
    print(f'{train_loss=:.3f}, state pred rel l2 error: {test_state_err:.3f}, cumulative reward pred abs error: {test_rew_err:.3f}')
    
    # wandb.log({"Train Loss": train_loss, 
    #             "": individal_losses[0], 
    #             "loss_sc": individal_losses[1], 
    #             "loss_grad": individal_losses[2],
    #              "loss_mu": individal_losses[3],
    #              "State Test Rel l2 Error": test_state_err,
    #              "State Test Rel l2 Error": state_err,
    #              "Cumulative Reward MSE": rew_err}, step=epoch)

        
    wandb.log({"Loss data term": train_loss, 
                 "State Test Rel l2 Error": test_state_err,
                 "State Train Rel l2 Error": state_err,
                 "Cumulative Reward MSE": rew_err,
                  "Cumulative Reward Test MSE": test_rew_err}, step=epoch)

wandb.finish()