import yaml


# Load new or existing config depending on training or validation/test run
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

    # Load existing config for validation/test run
    else:

        # Open existing run_info config and load parameters
        with open(f"{config_dir}/run_info.yaml", "r") as file:
            config = yaml.safe_load(file)

    return config


# Update run_info.yaml after a run
def save_run_info(run_info_path, run_info, rew_train, rew_val, rew_test):

    run_info["Train Reward"] = rew_train
    run_info["Validation Reward"] = rew_val
    run_info["Test Reward"] = rew_test

    with open(run_info_path, 'w') as file:
        yaml.dump(run_info, file, default_flow_style=False)