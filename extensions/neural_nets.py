import torch
import torch.nn as nn
import torch.optim as optim

def create_network(nn_architecture, state_shape, action_size, learning_rate):
    structure = nn_architecture['structure']
    if structure == 'DNN':
        network = DNN(nn_architecture, state_shape, action_size)
    elif structure == 'CNN':
        network = CNN(nn_architecture, state_shape, action_size)
    else:
        raise ValueError(f"Unsupported structure: {structure}")

    network.optimizer = optim.Adam(network.parameters(), lr = learning_rate)
    network.loss_fn = nn.MSELoss()

    print(f"Created {structure} Network Structure:\n{network}")

    return network

class DNN(nn.Module):
    def __init__(self, nn_architecture, state_shape, action_size):
        super(DNN, self).__init__()
        layers = []
        input_size = state_shape
        num_layers = nn_architecture['num_layers']
        num_neurons = nn_architecture['num_neurons']
        activation = nn_architecture['activation_layers']
        activation_output = nn_architecture['activation_output']

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(self.get_activation(activation))
            input_size = num_neurons

        layers.append(nn.Linear(input_size, action_size))

        if activation_output is not None:
            layers.append(self.get_activation(activation_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_activation(name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'linear':
            return nn.Identity()
        elif name == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

class CNN(nn.Module):
    def __init__(self, nn_architecture, state_shape, action_size):
        super(CNN, self).__init__()
        self.conv_layers = self.build_conv_layers(nn_architecture, state_shape)
        self.fc_layers = self.build_fc_layers(nn_architecture, action_size)

    def build_conv_layers(self, nn_architecture, state_shape):
        layers = []
        height, width, input_channels = state_shape

        num_conv_layers = nn_architecture['num_convolutional_layers']
        num_channels = nn_architecture['num_convolutional_channels']
        kernel_size = nn_architecture['kernel_size']
        stride = nn_architecture['stride']
        padding = nn_architecture['padding']
        activation = nn_architecture['activation_convolutional_layers']

        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(input_channels, num_channels, kernel_size, stride, padding))
            layers.append(self.get_activation(activation))
            input_channels = num_channels
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

        self.fc_input_size = num_channels * height * width
        return nn.Sequential(*layers)

    def build_fc_layers(self, nn_architecture, action_size):
        layers = []
        num_fc_layers = nn_architecture['num_fully_connected_layers']
        num_neurons = nn_architecture['num_neurons_fully_connected']
        activation = nn_architecture['activation_fully_connected_layers']
        activation_output = nn_architecture.get('activation_output', None)

        fc_input_size = self.fc_input_size
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(fc_input_size, num_neurons))
            layers.append(self.get_activation(activation))
            fc_input_size = num_neurons

        layers.append(nn.Linear(fc_input_size, action_size))
        if activation_output:
            layers.append(self.get_activation(activation_output))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    @staticmethod
    def get_activation(name):
        """Return the activation function based on its name."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'linear':
            return nn.Identity()
        elif name == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation function: {name}")
        
    @staticmethod
    def get_feature_size(nn_architecture, state_shape):
        height, width, channels = state_shape
        num_conv_layers = nn_architecture['num_convolutional_layers']
        kernel_size = nn_architecture['kernel_size']
        stride = nn_architecture['stride']
        padding = nn_architecture['padding']

        for _ in range(num_conv_layers):
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

        return height * width * nn_architecture['num_convolutional_channels']

class NoisyAdapter(nn.Module):
    def __init__(self, base_network, action_size):
        """
        Adapts a base Q-network into a noisy Q-network by replacing linear layers
        with noisy linear layers.

        Parameters:
            base_network (nn.Module): The existing Q-network to adapt.
            action_size (int): Number of actions in the environment.
        """
        super().__init__()
        self.base_network = base_network
        self.action_size = action_size

        if isinstance(base_network, DNN):
            noisy_layers = [
                NoisyLinear(layer.in_features, layer.out_features) if isinstance(layer, nn.Linear) else layer
                for layer in base_network.model.children()
            ]
            base_network.model = nn.Sequential(*noisy_layers)
        elif isinstance(base_network, CNN):
            noisy_fc_layers = [
                NoisyLinear(layer.in_features, layer.out_features) if isinstance(layer, nn.Linear) else layer
                for layer in base_network.fc_layers.children()
            ]
            base_network.fc_layers = nn.Sequential(*noisy_fc_layers)
        else:
            raise ValueError("Unsupported network type for noisy conversion.")

        self.optimizer = getattr(base_network, "optimizer", None)
        self.loss_fn = getattr(base_network, "loss_fn", None)

        print(f"Converted to Noisy Network with updated layers:\n{self}")

    def forward(self, x):
        """
        Forward method for noisy network, propagating prior logic.
        """
        x = self.base_network(x)

        # If the previous layer outputs a distribution, retain the format
        if hasattr(self.base_network, "num_atoms"):
            return x.view(-1, self.action_size, self.base_network.num_atoms)
        return x

    def reset_noise(self):
        """
        Reset noise in all noisy layers.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def __str__(self):
        """
        Override the string representation to print the updated noisy network structure.
        """
        return f"{self.base_network}"

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = 0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))


        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / (self.in_features**0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features**0.5))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features**0.5))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)
    
def convert_to_distributional_network(base_network, action_size, num_atoms):
    """
    Converts a base Q-network into a Distributional Q-network.

    Parameters:
        base_network (nn.Module): The existing Q-network to adapt.
        action_size (int): Number of actions in the environment.
        num_atoms (int): Number of atoms for the distributional output.
        v_min (float): Minimum value of the distributional support.
        v_max (float): Maximum value of the distributional support.

    Returns:
        nn.Module: A distributional Q-network.
    """
    class DistributionalAdapter(nn.Module):
        def __init__(self, base_network, action_size, num_atoms):
            super().__init__()
            self.base_network = base_network
            self.action_size = action_size
            self.num_atoms = num_atoms

            if isinstance(base_network, DNN):
                feature_layers = list(base_network.model[:-1])  # Remove last layer
                input_size = base_network.model[-1].in_features
            elif isinstance(base_network, CNN):
                feature_layers = base_network.fc_layers[:-1]  # Remove last fully connected layer
                input_size = base_network.fc_layers[-1].in_features
            else:
                raise ValueError("Unsupported network type for distributional conversion.")

            self.feature_layer = nn.Sequential(*feature_layers)
            self.output_layer = nn.Linear(input_size, action_size * num_atoms)

            self.optimizer = getattr(base_network, "optimizer", None)
            self.loss_fn = getattr(base_network, "loss_fn", None)

            print(f"Created Distributional Network with updated output layer \n{self.output_layer}\n")

        def forward(self, x):
            """
            Forward method for distributional network, propagating prior logic.
            """
            features = self.feature_layer(x)
            logits = self.output_layer(features).view(-1, self.action_size, self.num_atoms)
            return torch.softmax(logits, dim=-1)
        
        def __str__(self):
            """
            Override the string representation to print the updated noisy network structure.
            """
            return f"{self.base_network}"

    return DistributionalAdapter(base_network, action_size, num_atoms)


def create_dueling_network(base_network, action_size, nn_architecture):
    """
    Converts a base Q-network into a Dueling Q-network.

    Parameters:
        base_network (nn.Module): The existing Q-network to adapt.
        state_shape (int or tuple): State size or shape.
        action_size (int): Number of actions.
        nn_architecture (dict): Neural network architecture configuration.

    Returns:
        nn.Module: A dueling Q-network.
    """
    class DuelingAdapter(nn.Module):
        def __init__(self, base_network, action_size):
            super().__init__()
            self.base_network = base_network
            self.action_size = action_size
            self.num_neurons_stream = nn_architecture["num_neurons"]

            self.optimizer = getattr(base_network, "optimizer", None)
            self.loss_fn = getattr(base_network, "loss_fn", None)

            if isinstance(base_network, nn.Sequential):
                feature_layer = list(base_network.children())[:-1]
                input_layer = list(base_network.children())[-1]
            elif hasattr(base_network, "model"):
                feature_layer = list(base_network.model.children())[:-1]
                input_layer = list(base_network.model.children())[-1]
            else:
                raise ValueError("Unsupported base_network structure.")

            if not isinstance(input_layer, nn.Linear):
                raise ValueError(
                    "The base network's last layer must be a linear layer for dueling network adaptation."
                )

            self.feature_layer = nn.Sequential(*feature_layer)
            input_size = input_layer.in_features

            self.value_stream = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            self.advantage_stream = nn.Sequential(
             nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
            )

            print(f"Converted to Dueling Network with updated streams:\n{self}")

        def forward(self, x):
            """
            Forward method that integrates previous extension logic.
            """
            features = self.feature_layer(x)

            if features.size(-1) != self.value_stream[0].in_features:
                raise RuntimeError(
                    f"Feature size mismatch: expected {self.value_stream[0].in_features}, got {features.size(-1)}"
                )

            value = self.value_stream(features)
            advantages = self.advantage_stream(features)

            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

            if hasattr(self.base_network, "num_atoms"):
                q_values = q_values.view(-1, self.action_size, self.base_network.num_atoms)
                return torch.softmax(q_values, dim=-1)

            return q_values

        def __str__(self):
            """
            Override the string representation to print the updated dueling network structure.
            """
            return (
                f"Feature Layer: {self.feature_layer}\n"
                f"Value Stream: {self.value_stream}\n"
                f"Advantage Stream: {self.advantage_stream}\n"
            )

    return DuelingAdapter(base_network, action_size)
    