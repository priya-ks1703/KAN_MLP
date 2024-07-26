import numpy as np
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


class Evaluation:

    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.folder_id = "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = SummaryWriter(os.path.join(store_dir, self.folder_id))
        self.stats = stats

    def write_episode_data(self, episode, eval_dict):
        """
        Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
        e.g. eval_dict = {"loss" : 1e-4}
        """

        for k in eval_dict:
            assert k in self.stats
            self.summary_writer.add_scalar(k, eval_dict[k], global_step=episode)

        self.summary_writer.flush()

    def close_session(self):
        self.summary_writer.close()


def sort_models(models):
    def model_sort_key(model_name):
        if model_name == 'best_model.pt':
            return float('inf')  # Ensure 'best_model.pt' is at the end
        else:
            return int(model_name.split('_')[-1].split('.')[0])

    return sorted(models, key=model_sort_key)


def plot_model_performance(data, dist_type, out_folder, model_types=None, name=""):
    plt.figure(figsize=(10, 6))
    
    if model_types is None:
        model_types = list(data.keys())

    for model_type in model_types:
        models = data[model_type]
        model_names = sort_models(list(models.keys()))
        means = [models[model][dist_type]["mean"] for model in model_names]
        stds = [models[model][dist_type]["std"] for model in model_names]
        # filter name_{idx}.pt -> idx for all model names
        model_names = [model.split('_')[-1].split('.')[0] for model in model_names]
        model_names[-1] = 'best'
        plt.errorbar(model_names, means, yerr=stds, marker='.', label=model_type, capsize=2)
    
    plt.xlabel(f'Best Models for each {model_names[1]} epochs')
    plt.ylabel('Mean Episode Reward')
    plt.title(f'Mean Episode Reward ({dist_type.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if name != "":
        name = f"_{name}"

    plt.savefig(f"{out_folder}/mean_episode_reward_{dist_type}{name}.png")


def initialize_env_settings(env, percentage_range=0.2, out_of_distribution_percentage_range=0):
    default_gravity = 9.8
    default_masscart = 1.0
    default_masspole = 0.1
    default_total_mass = default_masspole + default_masscart
    default_length = 0.5  
    default_polemass_length = default_masspole * default_length
    default_force_mag = 10.0
    default_tau = 0.02 

    uniform_range = (1-percentage_range, 1+percentage_range)

    if out_of_distribution_percentage_range != 0:
        coin_toss = random.randint(0, 1)
        if coin_toss == 1:
            uniform_range = (1-percentage_range-out_of_distribution_percentage_range, 1-percentage_range)
        else:
            uniform_range = (1+percentage_range, 1+percentage_range+out_of_distribution_percentage_range)

    env.gravity = default_gravity * random.uniform(*uniform_range)
    env.masscart = default_masscart * random.uniform(*uniform_range)
    env.masspole = default_masspole * random.uniform(*uniform_range)
    env.total_mass = default_total_mass * random.uniform(*uniform_range)
    env.length = default_length * random.uniform(*uniform_range)
    env.pole_mass_length = default_polemass_length * random.uniform(*uniform_range)
    env.force_mag = default_force_mag * random.uniform(*uniform_range)
    env.tau = default_tau * random.uniform(*uniform_range)


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype("float32")


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
