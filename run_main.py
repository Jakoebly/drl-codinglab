import subprocess

args = [
    "./main.py",
    "--experiment_name", "test1",
    "--algorithm", "dqn",
    "--variant", "0",
    "--mode", "train",
    "--save",
    "--track",
]

subprocess.run(args)