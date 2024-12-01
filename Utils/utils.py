import matplotlib.pyplot as plt

def rewards_plotting(rew):
    x = range(len(rew))
    plt.figure(figsize=(12, 8))
    plt.plot(x, rew, linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.xticks(x)
    plt.show()