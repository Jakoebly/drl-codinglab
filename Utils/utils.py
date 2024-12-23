import yaml
import environment as env


# Config
def load_config(config_dir, algorithm):

    if config_dir == "../config":

        with open(f"{config_dir}/general.yaml", "r") as file:
            general = yaml.safe_load(file)

        #TODO: specify funtions in environment
        general["state_size"] = env.get_state_size()
        general["num_actions"] = env.get_num_actions()
        general["num_episodes"] = env.get_num_episodes()
        general["num_steps_trajectory"] = env.get_trajectory_length()

        with open(f"{config_dir}/hyperparams.yaml", "r") as file:
            hyperparams = yaml.safe_load(file).get(algorithm, {})

        config = general | hyperparams

    else:
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