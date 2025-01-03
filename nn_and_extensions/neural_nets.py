import torch
import torch.nn as nn
import torch.optim as optim

class DNN_Constructor(nn.Module):
    def __init__(self, nn_architecture):
        super().__init__()
        layers = []
        for i in range(len(nn_architecture) - 1):
            in_size, activation = nn_architecture[i]
            out_size, _ = nn_architecture[i + 1]

            layers.append(nn.Linear(in_size, out_size))

            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, last_in_size))

        if last_activation.lower() == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif last_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation.lower() == "linear" or last_activation.lower() is None:
            pass
        else:
            raise ValueError(f"Unsupported activation function: {last_activation}")

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class CNN_Constructor(nn.Module):
    print("test")









    
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class DistributionalQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units, num_atoms, v_min, v_max):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_units, action_size * num_atoms)

    def forward(self, x):
        features = self.feature_layer(x)
        logits = self.output_layer(features)
        logits = logits.view(-1, self.num_atoms)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# Noisy Q-Network
class NoisyQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super().__init__()
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_size, hidden_units),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units),
            nn.ReLU(),
            NoisyLinear(hidden_units, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units),
            nn.ReLU(),
            NoisyLinear(hidden_units, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
