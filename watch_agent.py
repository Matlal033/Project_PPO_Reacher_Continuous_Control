import sys
import torch
import numpy as np
from ActorCritic import ActorCritic
from unityagents import UnityEnvironment
from agent import Agent
from agent import collect_trajectories

if __name__ == "__main__":

    mean_score_tresh_to_save = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    episode = 2000
    discount_rate = .99
    tau = 0.95
    surrogate_clip = 0.2
    surrogate_clip_decay = 1 #0.999
    beta = 1e-2 #entropy coefficient
    beta_decay = 1 #0.998 #0.999
    tmax = 400000 #timesteps while collecting trajectories
    SGD_epoch = 4 #12 #11 #10 #Learning with the same trajectories (reshuffled, but still)
    LR = 1e-4
    adam_epsilon = 3e-4
    batch_size = 500 #1400 #2000
    hidden_size = 128  #500
    gradient_clip = 5 #2 #5
    rollout_size = 500

    env = UnityEnvironment(file_name='Reacher_Windows_x86_64\Reacher.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations #20 in parallel
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size

    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)      # initialize the score (for each agent)                     # list containing scores from each episode

    mean_rewards = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = Agent(action_size, state_size, device, episode=episode, discount_rate=discount_rate,
            tau=tau, surrogate_clip=surrogate_clip, beta=beta, tmax=tmax, SGD_epoch=SGD_epoch,
            LR=LR, adam_epsilon=adam_epsilon, batch_size=batch_size, hidden_size=hidden_size,
            num_agents=num_agents, gradient_clip=gradient_clip)

    #Load previously saved network weights
    try:
        filename = sys.argv[1]
    except:
        filename = None

    if filename:
        checkpointActorCritic = torch.load(filename)
        print('loading')
        agent.ActorCritic.load_state_dict(checkpointActorCritic)

        env_info = env.reset(train_mode=False)[brain_name] #Reset at each new episode
        states = env_info.vector_observations                  # get the current state (for each agent)                       # initialize the score (for each agent)
        dones = np.zeros((num_agents,1))
        while not np.any(dones):

            actions, log_probs_old, values = agent.select_action(states)
            env_info = env.step(actions.cpu().numpy())[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)

            dones = np.array([1 if d else 0 for d in env_info.local_done])#env_info.local_done                        # see if episode finished

            states = next_states # roll over states to next time step

        #Play episode with new policy and get score
        ep_mean_score = play_episode(agent, env, brain_name, num_agents)

        #Store the score of actual and the last 100 episodeS
        print('Episode ', ep+1,' avg score: ', ep_mean_score)
