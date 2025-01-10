import torch
from neural_nets import NoisyAdapter, DNN

# Define network input size and architecture
state_shape = 128
action_size = 10
nn_architecture = {
    "structure": "DNN",
    "num_layers": 3,
    "num_neurons": 128,
    "activation_layers": "relu",
    "activation_output": None,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize base network and noisy adapter
base_network = DNN(nn_architecture, state_shape, action_size)
noisy_network = NoisyAdapter(base_network, action_size).to(device)

# Generate random states and test Q-value distribution during training
print("Q-values during training (with noise):")
noisy_network.train()  # Set to training mode
for i in range(5):
    state = torch.randn(1, state_shape).to(device)  # Generate a random state
    q_values = noisy_network(state)  # Get Q-values
    print(f"State {i+1}: {state}")
    print(f"Q-values {i+1}: {q_values.cpu().detach().numpy()}\n")

# Generate random states and test Q-value distribution during evaluation
print("Q-values during evaluation (without noise):")
noisy_network.eval()  # Set to evaluation mode
for i in range(5):
    state = torch.randn(1, state_shape).to(device)  # Generate a random state
    q_values = noisy_network(state)  # Get Q-values
    print(f"State {i+1}: {state}")
    print(f"Q-values {i+1}: {q_values.cpu().detach().numpy()}\n")