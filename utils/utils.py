import yaml
import os
import csv
import numpy as np


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
        config["Avg. Train Reward"] = None
        config["Avg. Validation Reward"] = None
        config["Avg. Test Reward"] = None


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
        config["Avg. Train Reward"] = float(np.mean(rew_train))
        eps_dir[0] = f"{model_dir}/train_eps.csv"

    if rew_val is not None:
        config["Avg. Validation Reward"] = float(np.mean(rew_val))
        eps_dir[1] = f"{model_dir}/val_eps.csv"

    if rew_test is not None:
        config["Avg. Test Reward"] = float(np.mean(rew_test))
        eps_dir[2] = f"{model_dir}/test_eps.csv"

    print(f"eps_dir: {eps_dir}")


    # Write run info into the corresponding .yaml file
    with open(f"{model_dir}/run_info.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


    # Write episode data into the corresponding .csv file
    for mode, dir in enumerate(eps_dir):
        if dir is not None:
            print(f"Mode: {mode}, Dir: {dir}")
            with open(dir, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward"])
                for eps, reward in enumerate(rewards[mode], start=1):
                    print(f"Eps: {eps}, Reward: {reward}")
                    writer.writerow([eps, reward])