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
    
    print(f"Created {structure} Network Structure:\n{network}")
    print(f"Optimizer: {network.optimizer} with learning rate {learning_rate}")
    
    return network

class DNN(nn.Module):
    def __init__(self, nn_architecture, state_shape, action_size, learning_rate):
        super(DNN, self).__init__()
        layers = []
        #input_size = state_shape[0]
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
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_fn = nn.MSELoss()

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
    def __init__(self, nn_architecture, state_shape, action_size, learning_rate):
        super(CNN, self).__init__()
        self.conv_layers = self.build_conv_layers(nn_architecture, state_shape)
        self.fc_layers = self.build_fc_layers(nn_architecture, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

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
        #x = x.view(x.size(0), -1)
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

'''class CNN(nn.Module):
    def __init__(self, nn_architecture, state_shape, action_size, learning_rate):
        super(CNN, self).__init__()
        layers = []

        # Unpack state shape (Height, Width, Channels)
        height, width, input_channels = state_shape

        num_conv_layers = nn_architecture['num_convolutional_layers']
        num_channels = nn_architecture['num_convolutional_channels']
        kernel_size = nn_architecture['kernel_size']
        stride = nn_architecture['stride']
        padding = nn_architecture['padding']
        activation = nn_architecture['activation_layers']
        activation_output = nn_architecture.get('activation_output', None)

        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(input_channels, num_channels, kernel_size, stride, padding))
            layers.append(self.get_activation(activation))
            input_channels = num_channels
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

        self.fc_input_size = num_channels * height * width

        num_fc_layers = nn_architecture['num_fully_connected_layers']
        num_neurons = nn_architecture['num_neurons_fully_connected']

        for _ in range(num_fc_layers):
            layers.append(nn.Linear(self.fc_input_size, num_neurons))
            layers.append(self.get_activation(activation))
            self.fc_input_size = num_neurons

        layers.append(nn.Linear(self.fc_input_size, action_size))
        if activation_output is not None:
            layers.append(self.get_activation(activation_output))

        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass of the CNN.
        Input: x of shape (Batch, Height, Width, Channels)
        """
        x = x.permute(0, 3, 1, 2) 
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                x = x.contiguous()
                x = x.reshape(x.size(0), -1)
            x = layer(x)
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
            raise ValueError(f"Unsupported activation function: {name}")'''

'''class QNetwork(nn.Module):
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
    
    def forward(self, state):
        output = self.model(state)
        return output'''
