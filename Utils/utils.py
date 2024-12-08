import matplotlib.pyplot as plt

def rewards_plotting(rew):
    x = range(len(rew))
    plt.figure(figsize=(12, 8))
    plt.plot(x, rew, linestyle='-', color='blue', alpha=0.7, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards")
    
    plt.xticks(range(0, len(rew), 100), fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()