import argparse

def parse_args():
    """Specifies relevant command-line arguments."""

    # Create parser
    parser = argparse.ArgumentParser(description="")

    # Algorithm-related args
    parser.add_argument("--algorithm", type=str, help="Algorithm to use, choose from ('dqn', 'ppo', 'sac')")
    parser.add_argument("--variant", type=int, default=0, help="Problem variant to use, choose from (0 = default, 1 = added capacity, 2 = obstacles)")
    parser.add_argument("--mode", type=str, default="train", help="Mode of the run, choose from ('train', 'validate', 'test')")
    parser.add_argument("--experiment_name", type=str, nargs="?", help="Name of the experiment, will be added to the default run_name")
    parser.add_argument("--model_name", type=str, nargs="?", help="Name of the model to load weights for validation/testing, format: 'CL__{args.algorithm.upper()}__Var{args.variant}__{datetime.now}__{args.experiment_name}'")
    parser.add_argument("--best", action="store_true", help="If toggled, the current best model will be used for validation/testing")
    parser.add_argument("--save", action="store_true", default=True, help="If toggled, the run will be saved under './logs'")
    parser.add_argument("--sweep", type=bool, default=False, nargs="?", const=True, help="If toggled, a sweep will be initilized")

    # Cuda-related args
    parser.add_argument("--torch_deterministic", type=bool, default=True, nargs="?", const=True, help="If toggled, `torch.backends.cudnn.deterministic` will be set to 'False'")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True, help="If toggled, cuda will be enabled by default")

    # WandB-related args
    parser.add_argument("--track", type=bool, default=False, nargs="?", const=True, help="If toggled, the run will be tracked with WandB")
    parser.add_argument("--wandb_projectname", type=str, default="Coding_Lab", help="WandB's project name")
    parser.add_argument("--wandb_entity", type=str, default="gowiththetensorflow", help="Entity (team) of WandB's project")

    # Parse args
    args = parser.parse_args()

    return validate_args(args)

def validate_args(args):
    """Validates input args."""

    if args.algorithm.lower() not in ["dqn", "ppo", "sac"]:
        raise ValueError("Specify a valid algorithm ('dqn', 'ppo', 'sac')!")

    if args.variant not in [0, 1, 2]:
        raise ValueError("Specify a valid problem variant (0 = default, 1 = added capacity, 2 = obstacles)!")

    if args.mode is None or args.mode.lower() not in ["train", "validate", "test"]:
        raise ValueError("Specify a valid mode ('train', 'validate', 'test')!")

    if args.mode.lower() == "train" and args.model_name is not None:
        raise ValueError("Make sure arg 'model_name' is 'None' for mode 'train' as training creates a new model!")

    if args.mode.lower() == "train" and args.best:
        raise ValueError("Arg 'best' is not applicable to mode 'train'!")

    if args.mode.lower() in ["validate", "test"] and args.model_name is None and args.best is False:
        raise ValueError("Make sure you provide an existing model that can be used for validation/testing or specify '--best' to use the current best model!")

    if args.best and args.model_name is not None:
        raise ValueError(
            "Do not specify arg 'model_name' when using arg 'best' as the current best model is automatically used!")

    if args.mode.lower() == "train" and args.best:
        raise ValueError("Arg 'best' is not applicable to mode 'train'!")

    if args.sweep in ["validate", "test"]:
        raise ValueError("Specify mode as 'train' for sweeps")

    return args