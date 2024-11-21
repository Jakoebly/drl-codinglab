from SAC.sac import SAC_algorithm
from environment import Environment

env = Environment(variant=0, data_dir="./data")

SAC_algorithm(env=env, no_episodes=800, mode="training")
