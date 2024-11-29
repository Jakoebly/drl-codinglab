from pathlib import Path

CONFIG_NAME = 'dqn_default_config.json'
WEIGHTS_NAME = 'DQN_det_weights_FINAL.weights.h5'

CONFIG_PATH = Path(__file__).parent / "Configs" / CONFIG_NAME
WEIGHT_PATH = Path(__file__).parent / "Models" / WEIGHTS_NAME
RUNS_PATH = Path(__file__).parent / "Runs"