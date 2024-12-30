import os
import random
import csv
import torch
import yaml
import numpy as np


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
    if config_dir == "./config":

        # Open general config and load parameters
        with open(f"{config_dir}/general.yaml", "r") as file:
            general = yaml.safe_load(file).get("General", {})

        general["state_size"] = env.get_state_size()
        general["num_episodes_train"] = len(env.training_episodes)
        general["num_episodes_validate"] = len(env.validation_episodes)
        general["num_episodes_test"] = len(env.test_episodes)
        general["num_steps_trajectory"] = env.episode_steps

        # Open algorithm-specific config and load parameters
        with open(f"{config_dir}/hyperparams.yaml", "r") as file:
            hyperparams = yaml.safe_load(file).get(algorithm, {})

        # Concatenate general and algorithm-specific parameters to final config
        config = general | hyperparams

        # Create key/value pairs for reward tracking
        config["avg_train_reward"] = None
        config["avg_validation_reward"] = None
        config["avg_test_reward"] = None

        # Adjust nn_architecture
        config["nn_architecture"][0][0] = config["state_size"]
        config["nn_architecture"][-1][0] = config["num_actions"]


    # Load existing config for validation/test run
    else:

        # Open existing run_info config and load parameters
        with open(f"{config_dir}/run_info.yaml", "r") as file:
            config = yaml.safe_load(file)

    return config



# Update run_info.yaml after a run
def save_run_info(rewards, config, model_dir, agent, algorithm, variant, mode, run_name):

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
        config["avg_train_reward"] = float(np.mean(rew_train))
        eps_dir[0] = f"{model_dir}/train_eps.csv"

    if rew_val is not None:
        config["avg_validation_reward"] = float(np.mean(rew_val))
        eps_dir[1] = f"{model_dir}/val_eps.csv"

    if rew_test is not None:
        config["avg_test_reward"] = float(np.mean(rew_test))
        eps_dir[2] = f"{model_dir}/test_eps.csv"


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






