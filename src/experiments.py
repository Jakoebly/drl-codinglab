import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import wandb
import yaml
import seaborn as sns
from src import settings


def load_experiment_settings(): #TODO: load experiment settings from yaml file

    config_dir = settings.CONFIG_DIR

    with open(f"{config_dir}/experiment_settings.yaml", "r") as file:
        exp_set = yaml.safe_load(file).get("experiments", {})
    return exp_set

def set_exp_vars(env):

    exp_set = load_experiment_settings()

    if exp_set["heatmap"]:
        env.cell_visit_count = np.zeros((env.vertical_cell_count, env.horizontal_cell_count))
    if exp_set["item_distribution"]:
        env.item_dist = np.zeros((env.vertical_cell_count, env.horizontal_cell_count))
    if exp_set["KPIs"]["avg_distance_per_item"]:
        env.distance_per_item = []
        env.distance = 0
    if exp_set["KPIs"]["avg_time_per_item"]:
        env.remaining_time_per_item = []
    if exp_set["KPIs"]["item_completion_rate"]:
        env.item_count = 0
        env.item_completion_count = 0

def update_exp_vars(env, act, new_loc, new_items, num_new_items, locs, time, load_before, load_after):
    exp_set = load_experiment_settings()

    if act != 0:
        if exp_set["heatmap"]:
            if env.agent_loc != env.target_loc:
                env.cell_visit_count[env.agent_loc[0]][env.agent_loc[1]] += 1

        if exp_set["KPIs"]["avg_distance_per_item"]:
            if act != 0:
                if new_loc in env.eligible_cells:
                    env.distance += 1
                    if env.agent_loc == env.target_loc:
                        if load_after > 0:
                            for _ in range(load_after):
                                env.distance_per_item.append(env.distance)
                            env.distance = 0

    if (load_before < env.agent_capacity) and (env.agent_loc in locs):
        if exp_set["KPIs"]["avg_time_per_item"]:
            env.remaining_time_per_item.append(env.max_response_time - time)
        if exp_set["KPIs"]["item_completion_rate"]:
            env.item_completion_count += 1

    if exp_set["KPIs"]["avg_time_per_item"]:
        env.item_count += num_new_items
    if exp_set["item_distribution"]:
        for item in new_items:
            env.item_dist[item[0]][item[1]] += 1

def get_cell_visit_count(env):
    return env.cell_visit_count

def save_cell_visit_count(env):
    exp_set = load_experiment_settings()
    if exp_set["heatmap"]:
        old_cell_visit_count = get_cell_visit_count(env).copy()
        return old_cell_visit_count

def set_cell_visit_count(env, cell_visit_count):
    env.cell_visit_count = cell_visit_count

def reset_cell_visit_count(env):
    exp_set = load_experiment_settings()

    if exp_set["heatmap"]:
        env.cell_visit_count = np.zeros((env.vertical_cell_count, env.horizontal_cell_count))

def log(env, mode, training, eval, episode, num_episodes):

    exp_set = load_experiment_settings()

    if not eval:
        if not training and episode == num_episodes - 1:
            log_kpis(env, exp_set)
        if (episode + 1) % exp_set["heatmap_log_interval"] == 0 and (episode + 1) >= exp_set["heatmap_log_interval"]:
            log_heatmap(env, exp_set, mode, episode)


def log_kpis(env, exp_set):

    if exp_set["KPIs"]["avg_distance_per_item"]:
        wandb.log({"KPIs/Avg. Distance per Item": np.mean(env.distance_per_item)})
    if exp_set["KPIs"]["avg_time_per_item"]:
        wandb.log({"KPIs/Avg. Remaining Time per Item": np.mean(env.remaining_time_per_item)})
    if exp_set["KPIs"]["item_completion_rate"]:
        wandb.log({"KPIs/Item Completion Rate": env.item_completion_count / env.item_count})

def log_heatmap(env, exp_set, mode, episode):
    if exp_set["heatmap"]:
        cell_visit_count_norm = env.cell_visit_count / np.max(env.cell_visit_count)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cell_visit_count_norm, annot=False, fmt=".0f", cmap="coolwarm")
        plt.title(f"Episode: {episode}", fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        heatmap = Image.open(buf).copy()
        buf.close()

        if mode == "training":
            wandb.log({"Heatmap_Train": wandb.Image(heatmap)}, episode)
        if mode == "validation":
            step = wandb.config["general"]["num_episodes_train"] + episode
            wandb.log({"Heatmap_Val": wandb.Image(heatmap)}, step)
        if mode == "testing":
            step = wandb.config["general"]["num_episodes_train"] + episode
            wandb.log({"Heatmap_Test": wandb.Image(heatmap)}, step)

        plt.close()

def log_item_dist(env):
    exp_set = load_experiment_settings()

    if exp_set["item_distribution"]:
        item_dist_norm = env.item_dist / np.max(env.item_dist)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(item_dist_norm, annot=False, fmt=".0f", cmap="coolwarm")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        heatmap = Image.open(buf).copy()
        buf.close()
        wandb.log({"Item Distribution": wandb.Image(heatmap)})

def set_render_flag(env, mode, eval, episode):

    exp_set = load_experiment_settings()

    render_flag = False if eval else exp_set["render"]
    env.render_enabled = False

    if render_flag:
        render_frames = {
            "training": exp_set["render_frames_train"],
            "validation": exp_set["render_frames_val"],
            "testing": exp_set["render_frames_val"]
        }
        if mode in render_frames and (episode + 1) in render_frames[mode]:
            print("Rendering Environment")
            env.render_enabled = True

def log_render(env, mode, episode):
    exp_set = load_experiment_settings()

    # Render logging
    if env.render_enabled:
        images = env.renderer.get_images()
        log_keys = {
            "training": [exp_set["render_frames_train"], "train_video"],
            "validation": [exp_set["render_frames_val"], "val_video"],
            "testing": [exp_set["render_frames_val"], "test_video"]
        }
        if mode in log_keys and (episode + 1) in log_keys[mode][0]:
            wandb.log({log_keys[mode][1]: wandb.Video(images, fps=24, format="mp4")})


def create_boxplot(algorithm):
    BEST_DIR = settings.BEST_DIR
    LOG_DIR = "./logs"

    with open(BEST_DIR, "r") as file:
        best_models = yaml.safe_load(file)
        var0_model_name = best_models["Variant_0"][algorithm.upper()]["model_name"]
        var1_model_name = best_models["Variant_1"][algorithm.upper()]["model_name"]
        var2_model_name = best_models["Variant_2"][algorithm.upper()]["model_name"]

    test_rews = np.zeros((3, 100))

    test_rews[0] = np.loadtxt(f"{LOG_DIR}/Variant_0/{algorithm.upper()}/{var0_model_name}/test_eps.csv", delimiter=",", skiprows=1)[:, 1]
    test_rews[1] = np.loadtxt(f"{LOG_DIR}/Variant_1/{algorithm.upper()}/{var1_model_name}/test_eps.csv", delimiter=",", skiprows=1)[:, 1]
    test_rews[2] = np.loadtxt(f"{LOG_DIR}/Variant_2/{algorithm.upper()}/{var2_model_name}/test_eps.csv", delimiter=",", skiprows=1)[:, 1]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(test_rews.T,
                      patch_artist=True,
                      labels=["Variant 0", "Variant 1", "Variant 2"],
                      boxprops=dict(facecolor="lightgray"),
                      medianprops=dict(color='black')
                      )

    plt.ylabel("Rewards")
    plt.grid(visible=True, axis='y')

    # Show the plot
    plt.show()




























