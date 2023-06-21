# Project Report

### Learning Algorithm

The PPO algorithm was used to solve this project.
It is an on-policy algorithm, from the family of policy gradient methods, that is able to tackle complex environments, continuous action spaces and distributed training.

One of the key element characterizing the PPO algorithm is the surrogate clipped objective. \
To train the neural net, we compare the probabilities of taking a given action according to a previous policy, versus the probabilities of taking that same action with the latest policy by dividing them together. \
Then, we clip the ratio of probabilities within a range close to one (e.g. [0.8 to 1,2]), and we multiply by the advantages of doing that action in that state space to obtain the clipped objective. \
Clipping the ratio decreases the steps while updating the policy, and makes it more stable. We compare that clipped objective (clipped ratio * advantage) to the unclipped objective (ratio * advantages), and the smallest value of the two is the one used in order to stay more conservative with the policy updates.

##### Hypermarameters :
    episode = 2000
    discount_rate = .99
    gae_lambda = 0.95
    surrogate_clip = 0.2
    surrogate_clip_decay = 0.999
    beta = 1e-2 #entropy coefficient
    beta_decay = 0.995
    tmax = 400000 #timesteps while collecting trajectories
    SGD_epoch = 10 
    LR = 3e-4
    adam_epsilon = 1e-5
    batch_size = 2000
    hidden_size = 1024
    gradient_clip = 5 
    rollout_size = 2000

##### Actor neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 1024 | ReLU |
|2  | Fully Connected | 1024 | 1024 | ReLU |
|3  | Fully Connected | 1024 | 4 (action size) | tanh |

##### Value neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 1024 | ReLU |
|2  | Fully Connected | 1024 | 1024 | ReLU |
|3  | Fully Connected | 1024 | 1 | None |

### Plot of rewards


### Ideas for future work

