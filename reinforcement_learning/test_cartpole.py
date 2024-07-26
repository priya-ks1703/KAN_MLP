import os
from datetime import datetime
import gym
import json
import torch
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
from agent.utils import plot_model_performance
import numpy as np
from fastkan import FastKAN as KAN
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    main_folder = "./models_cartpole"
    n_test_episodes = 10

    total_results = {}

    for model_folder in os.listdir(main_folder):
        # continue if not a folder:
        if not os.path.isdir(f"{main_folder}/{model_folder}"):
            continue
        print(f"Testing model {model_folder} ...")
        # load config json
        with open(f"{main_folder}/{model_folder}/config.json", "r") as f:
            model_config = json.load(f)
        
        if model_config[0] == 'kan':
            Q_network = KAN(model_config[1], num_grids=5).to(device)
            Q_target = KAN(model_config[1], num_grids=5).to(device)
        elif model_config[0] == 'mlp':
            Q_network = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dims=model_config[1]).to(device)
            Q_target = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dims=model_config[1]).to(device)
        else:
            raise ValueError("Model not found")
        
        agent = DQNAgent(Q_network, Q_target, num_actions=4, device=device)

        results = {}

        for model in os.listdir(f"{main_folder}/{model_folder}"):
            if not model.endswith(".pt"):
                continue

            print(f"Testing checkpoint {model} ...")

            agent.load(f"{main_folder}/{model_folder}/{model}")

            results[model] = {}

            for dist_type in [('in_distribution', model_config[-2], model_config[-1]), 
                              # model_config[-1] contains out of distribution range (0 by default)
                              ('out_of_distribution', model_config[-2], model_config[-1]+0.2)]:
                episode_rewards = []

                for i in range(n_test_episodes):
                    stats = run_episode(env, 
                                        agent, 
                                        deterministic=True, 
                                        do_training=False, 
                                        rendering=False,
                                        in_distribution_range=dist_type[1], 
                                        out_distribution_range=dist_type[2])
                    episode_rewards.append(stats.episode_reward)

                # save results in a dictionary and write them into a .json file
                model_results = dict()
                model_results["episode_rewards"] = episode_rewards
                model_results["mean"] = np.array(episode_rewards).mean()
                model_results["std"] = np.array(episode_rewards).std()

                results[model][dist_type[0]] = model_results

        with open(f"{main_folder}/{model_folder}/results.json", "w") as f:
            json.dump(results, f, indent=4)

        total_results[model_folder] = results

    with open(f"{main_folder}/total_results.json", "w") as f:
        json.dump(total_results, f, indent=4)

    env.close()
    print("... finished")
