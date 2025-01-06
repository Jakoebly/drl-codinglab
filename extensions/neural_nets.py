import torch
import torch.nn as nn
import torch.optim as optim

def create_network(nn_architecture, state_shape, action_size, learning_rate):
    structure = nn_architecture['structure']
    if structure == 'DNN':
        network = DNN(nn_architecture, state_shape, action_size, learning_rate)
    elif structure == 'CNN':
        network = CNN(nn_architecture, state_shape, action_size, learning_rate)
    else:
        raise ValueError(f"Unsupported structure: {structure}")

    network.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
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

            print(f"Created Distributional Network with updated output layer \n{self.output_layer}")

        def forward(self, x):
            features = self.feature_layer(x)
            logits = self.output_layer(features).view(-1, self.action_size, self.num_atoms)
            return torch.softmax(logits, dim=-1)

    return DistributionalAdapter(base_network, action_size, num_atoms)

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        self.feature_layer = DNN(input_dim, hidden_units)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DuelingQNetworkCNN(nn.Module):
    def __init__(self, nn_architecture, state_shape, action_size):
        super().__init__()
        self.conv_layers = CNN.build_conv_layers(nn_architecture, state_shape)
        feature_size = CNN.get_feature_size(nn_architecture, state_shape)

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, nn_architecture["num_neurons"]),
            nn.ReLU(),
            nn.Linear(nn_architecture["num_neurons"], 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, nn_architecture["num_neurons"]),
            nn.ReLU(),
            nn.Linear(nn_architecture["num_neurons"], action_size),
        )

    def forward(self, x):
        features = self.conv_layers(x).flatten(start_dim=1)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class NoisyQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        self.feature_layer = DNN(input_dim, hidden_units, noisy=True)
        self.output_layer = NoisyLinear(hidden_units, output_dim)

    def forward(self, x):
        features = self.feature_layer(x)
        q_values = self.output_layer(features)
        return q_values