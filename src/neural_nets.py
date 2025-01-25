import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.modules import NoisyLinear


def build_network(nn_architecture, state_size, action_size, learning_rate):
    structure = nn_architecture["structure"]
    if structure == "DNN":
        network = DNN(nn_architecture, state_size, action_size)
    elif structure == "CNN":
        network = CNN(nn_architecture, state_size, action_size)
    else:
        raise ValueError(f"Unsupported structure: {structure}")

    network.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    network.loss_fn = nn.MSELoss()

    print(network)

    return network


class DNN(nn.Module):
    def __init__(self, nn_architecture, state_size, action_size):
        super(DNN, self).__init__()

        layers = []
        input_size = state_size
        num_layers = nn_architecture["num_layers"]
        num_neurons = nn_architecture["num_neurons"]
        activation_layers = nn_architecture["activation_layers"]
        activation_output = nn_architecture["activation_output"]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(self.get_activation(activation_layers))
            input_size = num_neurons

        layers.append(nn.Linear(input_size, action_size))

        if activation_output is not None:
            layers.append(self.get_activation(activation_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_activation(name):
        """
        Return the activation function based on its name.
        """
        activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "linear": nn.Identity(), "softmax": nn.Softmax(dim=1)}

        if name not in activations:
            raise ValueError(f"Unsupported activation function: {name}")

        return activations[name]


class CNN(nn.Module):
    def __init__(self, nn_architecture, state_size, action_size):
        super(CNN, self).__init__()

        grid_state_size, scalar_state_size = state_size
        self.conv_layers = self.build_conv_layers(nn_architecture, grid_state_size)
        
        # Adding a fully connected layer to process convolutional output
        self.num_neurons_fully_feature_input = nn_architecture["num_neurons_fully_feature_input"]
        self.feature_fc = nn.Linear(self.fc_input_size, self.num_neurons_fully_feature_input)
        self.feature_activation = self.get_activation(nn_architecture["activation_fully_connected_layers"])
        
        # Update input size for fully connected layers to include scalar inputs
        fc_input_size = self.num_neurons_fully_feature_input + scalar_state_size[0]
        self.fc_layers = self.build_fc_layers(nn_architecture, fc_input_size, action_size)

    def build_conv_layers(self, nn_architecture, grid_state_size):
        """
        Constructs the convolutional layers.

        Parameters:
            nn_architecture (dict): Configuration for convolutional layers.
            grid_state_size (tuple): Shape of the grid state (Height, Width, Channels).

        Returns:
            nn.Sequential: Convolutional layers.
        """
        layers = []
        height, width, input_channels = grid_state_size

        num_conv_layers = nn_architecture["num_convolutional_layers"]
        num_channels = nn_architecture["num_convolutional_channels"]
        kernel_size = nn_architecture["kernel_size"]
        stride = nn_architecture["stride"]
        padding = nn_architecture["padding"]
        activation = nn_architecture["activation_convolutional_layers"]

        for _ in range(num_conv_layers):
            # Calculate 'same' padding
            padding = (kernel_size - 1) // 2 if stride == 1 else \
                    (stride - 1) * height - stride + kernel_size // 2
            layers.append(nn.Conv2d(input_channels, num_channels, kernel_size, stride, padding))
            layers.append(self.get_activation(activation))
            input_channels = num_channels
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

        self.fc_input_size = num_channels * height * width

        return nn.Sequential(*layers)

    def build_fc_layers(self, nn_architecture, fc_input_size, action_size):
        """
        Constructs the fully connected layers.

        Parameters:
            nn_architecture (dict): Configuration for fully connected layers.
            fc_input_size (int): Input size for the first fully connected layer.
            action_size (int): Number of actions in the environment.

        Returns:
            nn.Sequential: Fully connected layers.
        """
        layers = []
        num_fc_layers = nn_architecture["num_fully_connected_layers"]
        num_neurons = nn_architecture["num_neurons_fully_connected"]
        activation = nn_architecture["activation_fully_connected_layers"]
        activation_output = nn_architecture.get("activation_output", None)

        for _ in range(num_fc_layers):
            layers.append(nn.Linear(fc_input_size, num_neurons))
            layers.append(self.get_activation(activation))
            fc_input_size = num_neurons

        layers.append(nn.Linear(fc_input_size, action_size))
        if activation_output:
            layers.append(self.get_activation(activation_output))

        return nn.Sequential(*layers)

    def forward(self, grid_state, scalar_state):
        """
        Forward pass through the CNN.

        Parameters:
            grid_state (torch.Tensor): Grid state tensor of shape (Batch, Height, Width, Channels).
            scalar_state (torch.Tensor): Scalar state tensor of shape (Batch, ScalarSize).

        Returns:
            torch.Tensor: Output of the network.
        """
        # Process grid state through convolutional layers
        grid_state = grid_state.permute(0, 3, 1, 2)
        conv_out = self.conv_layers(grid_state).flatten(1)

        # Process convolutional output through the fully connected feature layer
        feature_out = self.feature_activation(self.feature_fc(conv_out))

        # Concatenate processed convolutional features with scalar inputs
        combined_input = torch.cat((feature_out, scalar_state), dim=1)

        # Process through the fully connected layers
        x = self.fc_layers(combined_input)

        return x

    @staticmethod
    def get_activation(name):
        """
        Return the activation function based on its name.
        """
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "linear": nn.Identity(),
            "softmax": nn.Softmax(dim=1),
        }

        if name not in activations:
            raise ValueError(f"Unsupported activation function: {name}")

        return activations[name]

    @staticmethod
    def get_feature_size(nn_architecture, grid_state_size):
        """
        Computes the flattened feature size after the convolutional layers.

        Parameters:
            nn_architecture (dict): Configuration for convolutional layers.
            grid_state_size (tuple): Shape of the grid state (Height, Width, Channels).

        Returns:
            int: Flattened feature size after the convolutional layers.
        """
        height, width, channels = grid_state_size
        num_conv_layers = nn_architecture["num_convolutional_layers"]
        kernel_size = nn_architecture["kernel_size"]
        stride = nn_architecture["stride"]
        padding = nn_architecture["padding"]

        for _ in range(num_conv_layers):
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

        return height * width * nn_architecture["num_convolutional_channels"]


class NoisyAdapter(nn.Module):
    def __init__(self, base_network, action_size, std_init, noisy_layers):
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
            new_layers = [layer for layer in self.base_network.model.children()]

            for i in range(1, 2 * noisy_layers):  # every second layer is an actual NN layer
                if isinstance(new_layers[-i], nn.Linear):  # start with last layer
                    new_layers[-i] = NoisyLinear(new_layers[-i].in_features, new_layers[-i].out_features, std_init=std_init)

            base_network.model = nn.Sequential(*new_layers)

        elif isinstance(base_network, CNN):
            new_fc_layers = [layer for layer in self.base_network.fc_layers.children()]

            # TODO: not tested yet!!!
            for i in range(1, 2 * noisy_layers):  # every second layer is an actual NN layer
                if isinstance(new_layers[-i], nn.Linear):  # start with last layer
                    new_fc_layers[-i] = NoisyLinear(new_fc_layers[-i].in_features, new_fc_layers[-i].out_features, std_init=std_init)

            base_network.fc_layers = nn.Sequential(*new_fc_layers)

        else:
            raise ValueError("Unsupported network type for noisy conversion.")

        self.optimizer = getattr(base_network, "optimizer", None)
        self.loss_fn = getattr(base_network, "loss_fn", None)

        print(f"Converted to Noisy Network with updated layers:\n{self}")

    def forward(self, x):
        """
        Forward method for noisy network, propagating prior logic.
        """
        # x = self.base_network(x)
        return self.base_network.model(x)

    def reset_noise(self):
        """
        Reset noise in all noisy layers.
        """
        for layer in self.base_network.model.children():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def __str__(self):
        """
        Override the string representation to print the updated noisy network structure.
        """
        return f"{self.base_network}"


def convert_to_distributional_network(base_network, action_size, num_atoms, v_min, v_max):
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
        def __init__(self, base_network, action_size, num_atoms, v_min, v_max):
            super().__init__()
            self.base_network = base_network
            self.action_size = action_size
            self.num_atoms = num_atoms
            self.v_min = v_min
            self.v_max = v_max

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
            probs = self.create_dist(x)
            return torch.sum(probs * torch.linspace(self.v_min, self.v_max, self.num_atoms), dim=2)
        
        def create_dist(self, x):
            features = self.feature_layer(x)
            logits = self.output_layer(features).view(-1, self.action_size, self.num_atoms)
            return torch.softmax(logits, dim=-1)


        def __str__(self):
            """
            Override the string representation to print the updated noisy network structure.
            """
            return f"{self.base_network}"

    return DistributionalAdapter(base_network, action_size, num_atoms, v_min, v_max)


def create_dueling_network(base_network, action_size, extensionparameters):
    """
    Converts a base Q-network into a Dueling Q-network.

    Parameters:
        base_network (nn.Module): The existing Q-network to adapt.
        action_size (int): Number of actions.
        extensionparameters (dict): Neural network architecture configuration.

    Returns:
        nn.Module: A dueling Q-network.
    """

    class DuelingAdapter(nn.Module):
        def __init__(self, base_network, action_size):
            super().__init__()
            self.base_network = base_network
            self.action_size = action_size

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
                raise ValueError("The base network's last layer must be a linear layer for dueling network adaptation.")

            self.feature_layer = nn.Sequential(*feature_layer)
            input_size = input_layer.in_features

            num_layers_stream = extensionparameters["dueling"]["num_layers_stream"]
            num_neurons_stream = extensionparameters["dueling"]["num_neurons_stream"]

            self.value_stream = self._create_stream(input_size, num_neurons_stream, num_layers_stream, 1)
            self.advantage_stream = self._create_stream(input_size, num_neurons_stream, num_layers_stream, action_size)

            print(f"Converted to Dueling Network with updated streams:\n{self}")

        @staticmethod
        def _create_stream(input_size, num_neurons, num_layers, output_size):
            """
            Creates a stream (value or advantage) with the specified number of layers and neurons.

            Parameters:
                input_size (int): Input size of the first layer.
                num_neurons (int): Number of neurons in each hidden layer.
                num_layers (int): Number of hidden layers.
                output_size (int): Size of the output layer.

            Returns:
                nn.Sequential: A neural network stream.
            """
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Linear(input_size, num_neurons))
                layers.append(nn.ReLU())
                input_size = num_neurons

            layers.append(nn.Linear(input_size, output_size))
            return nn.Sequential(*layers)

        def forward(self, x):
            """
            Forward method that integrates previous extension logic.
            """
            features = self.feature_layer(x)

            if features.size(-1) != self.value_stream[0].in_features:
                raise RuntimeError(f"Feature size mismatch: expected {self.value_stream[0].in_features}, got {features.size(-1)}")

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
            return f"Feature Layer: {self.feature_layer}\n" f"Value Stream: {self.value_stream}\n" f"Advantage Stream: {self.advantage_stream}\n"

    return DuelingAdapter(base_network, action_size)
