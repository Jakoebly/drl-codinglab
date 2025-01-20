import os
import random
import csv
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src import settings

# Set seeds
def set_seeds(seed, torch_deterministic):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

# Load new or existing config depending on training or validation/testing
def load_config(env, config_dir, algorithm):

    # Load config for training run
    if config_dir == settings.CONFIG_DIR:

        # Open algorithm-specific config and load parameters
        with open(f"{config_dir}/hyperparams.yaml", "r") as file:
            hyperparams = yaml.safe_load(file).get(algorithm, {})

        # Adjust config with sections 'general' and 'results'
        config = hyperparams
        config["general"] = {}
        config["results"] = {}
        config["features"] = {}

        # Open feature-specific config and load parameters
        with open(f"{config_dir}/feature_config.yaml", "r") as file:
            feature_config = yaml.safe_load(file).get("obs", {})
            config["features"] = feature_config
            env.obs_config = feature_config

        if algorithm == "DQN":
            config["general"]["state_size"] = env.get_state_size(hyperparams["nn_architecture"]["structure"])
        elif algorithm == "PPO":
            config["general"]["state_size"] = env.get_state_size(hyperparams["nn_architecture_actor"]["structure"])
        elif algorithm == "SAC":
            pass

        config["general"]["action_size"] = env.action_size
        config["general"]["num_episodes_train"] = len(env.training_episodes)
        config["general"]["num_episodes_validate"] = len(env.validation_episodes)
        config["general"]["num_episodes_test"] = len(env.test_episodes)
        config["general"]["num_steps_trajectory"] = env.episode_steps

        config["results"]["avg_train_reward"] = None
        config["results"]["avg_validation_reward"] = None
        config["results"]["avg_test_reward"] = None

    # Load existing config for validation/test run
    else:
        # Open existing run_info config and load parameters
        with open(f"{config_dir}/run_info.yaml", "r") as file:
            config = yaml.safe_load(file)

    return config

def load_sweep_config(config_dir, sweep_name, algorithm):

    with open(f"{config_dir}/sweeps/{algorithm.lower()}_sweep_config.yaml", "r+") as file:
        sweep_config = yaml.safe_load(file)
        sweep_config["name"] = sweep_name

    return sweep_config

def update_env_features(env, config, algorithm):

    env.obs_config = config["features"]
    if algorithm == "DQN":
        config["general"]["state_size"] = env.get_state_size(config["nn_architecture"]["structure"])
    elif algorithm == "PPO":
        config["general"]["state_size"] = env.get_state_size(config["nn_architecture_actor"]["structure"])
    elif algorithm == "SAC":
        pass

# Update run_info.yaml after a run
def save_run_info(rewards, config, model_dir, agent, algorithm, variant, mode, run_name, wandb_run):

    # If training: create new subfolder and save weights
    if mode == "train":
        model_dir = f"./logs/Variant_{variant}/{algorithm}/{run_name}"
        os.makedirs(model_dir, exist_ok=True)

        agent.save_weights(f"{model_dir}/model.pt")

    # Extract rewards
    rew_train = rewards[0]
    rew_val = rewards[1]
    rew_test = rewards[2]


    eps_dir = [None, None, None]

    # Update relevant key/value pairs and create/reference .csv files for reward tracking
    if rew_train is not None:
        eps_dir[0] = f"{model_dir}/train_eps.csv"
    if rew_val is not None:
        eps_dir[1] = f"{model_dir}/val_eps.csv"
    if rew_test is not None:
        eps_dir[2] = f"{model_dir}/test_eps.csv"

    wandb_run.config.update({
        "results": {
            "avg_train_reward": float(np.mean(rew_train)) if rew_train else None,
            "avg_validation_reward": float(np.mean(rew_val)) if rew_val else None,
            "avg_test_reward": float(np.mean(rew_test)) if rew_test else None
        }
    }, allow_val_change=True)


    # Write run info into the corresponding .yaml file
    with open(f"{model_dir}/run_info.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


    # Write episode data into the corresponding .csv file
    for mode, dir in enumerate(eps_dir):
        if dir is not None:
            with open(dir, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward"])
                for eps, reward in enumerate(rewards[mode], start=1):
                    writer.writerow([eps, reward])

def load_best(best_dir, variant, algorithm):
    with open(best_dir, 'r') as file:
        model_name = yaml.safe_load(file).get(f"Variant_{variant}", {}).get(algorithm, {}).get("model_name", {})
    return model_name

def update_best(algorithm, variant, rewards, model_name, run_name, best_dir):
    # Extract validation rewards
    rew_val = rewards[1]

    # Check if validation rewards exist
    if rew_val is not None:

        # Check if the run was an integrated train/validation run and adjust naming
        if model_name is None:
            model_name = run_name

        # Open best.yaml file
        with open(best_dir, 'r') as file:
            content = yaml.safe_load(file)

        # Extract relevant section from the content of the best.yaml file
        best = content[f"Variant_{variant}"].get(algorithm, {})

        # Adjust relevant parameters if current model is better than the current best model
        if best["avg_validation_reward"] < float(np.mean(rew_val)):
            best["model_name"] = model_name
            best["avg_validation_reward"] = float(np.mean(rew_val))

        # Update the best.yaml content with new data
        content[f"Variant_{variant}"][algorithm] = best

        # Update the best.yaml file
        with open(best_dir, 'w') as file:
            yaml.safe_dump(content, file, default_flow_style=False, sort_keys=False)

def visualize_prioritized_replay(agent, num_samples=1000):
    # Retrieve priorities from the buffer
    if not hasattr(agent.buffer, 'priorities'):
        print("Prioritized Replay not enabled!")
        return
    
    priorities = np.array(agent.buffer.priorities)
    if len(priorities) == 0:
        print("Replay buffer is empty!")
        return
    
    # Normalize priorities for sampling probabilities
    sampling_probs = priorities / priorities.sum()

    # Sample indices and compute IS weights
    indices = np.random.choice(len(priorities), num_samples, p=sampling_probs)
    sampled_priorities = priorities[indices]
    weights = (len(priorities) * sampling_probs[indices]) ** -agent.beta
    weights /= weights.max()  # Normalize IS weights

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Priority Distribution
    axes[0].hist(priorities, bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title("Priority Distribution")
    axes[0].set_xlabel("Priority")
    axes[0].set_ylabel("Count")

    # Sampling Probability vs Priority
    axes[1].scatter(priorities, sampling_probs, alpha=0.6, color='orange')
    axes[1].set_title("Sampling Probability vs Priority")
    axes[1].set_xlabel("Priority")
    axes[1].set_ylabel("Sampling Probability")

    # Importance Sampling Weights
    axes[2].hist(weights, bins=50, color='green', edgecolor='black')
    axes[2].set_title("Importance Sampling Weights")
    axes[2].set_xlabel("Weight")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.show()





