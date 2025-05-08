



Discrete Neural SDE training is done [here](./train_nsde_deterministic.py), by default on the inverted pendulum dataset. One can change this to the swimmer dataset by uncommenting line 11 and line 335. 
The analogous thing can be done for the probabalistic training [here](./train_nsde_deterministic.py). [train_nsde_deterministic.py](./train_nsde_deterministic.py) has comments and referneces to the equations in the paper where the code implements them. 

The A2C actor critic algorithm is implemented in [actor_critic_inverted_pendulum.py](./actor_critic_inverted_pendulum.py)