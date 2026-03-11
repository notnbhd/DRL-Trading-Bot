import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Actor network using CNN-LSTM architecture"""

    def __init__(self, input_shape, action_space):
        """
        Parameters
        ----------
        input_shape : tuple
            (lookback_window, features)
        action_space : int
            Number of possible actions
        """
        super().__init__()
        seq_len, n_features = input_shape

        # Feature Learning — Conv1D expects (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=32,
            kernel_size=3,
            padding=1,  # 'same' padding
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Sequence Learning
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
        )

        # Decision
        self.fc1 = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, action_space)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, features)

        Returns
        -------
        torch.Tensor
            Action probabilities, shape (batch, action_space)
        """
        # Conv1D needs (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # LSTM needs (batch, seq_len, features)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)  # (batch, 32)

        x = self.relu(self.fc1(x))
        x = torch.softmax(self.fc_out(x), dim=-1)
        return x


class CriticNetwork(nn.Module):
    """Critic network using CNN-LSTM architecture"""

    def __init__(self, input_shape):
        """
        Parameters
        ----------
        input_shape : tuple
            (lookback_window, features)
        """
        super().__init__()
        seq_len, n_features = input_shape

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
        )

        self.fc1 = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, features)

        Returns
        -------
        torch.Tensor
            State value, shape (batch, 1)
        """
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)

        x = self.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


class CNNLSTM:
    """CNN-LSTM model for feature extraction and time series forecasting based on the paper architecture"""

    def __init__(self, input_shape, action_space, learning_rate=0.00025):
        """
        Initialize the CNN-LSTM model

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (lookback_window, features)
        action_space : int
            Number of possible actions
        learning_rate : float, optional
            Learning rate for the Adam optimizer
        """
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate

        # Build actor and critic models
        self.actor = ActorNetwork(input_shape, action_space)
        self.critic = CriticNetwork(input_shape)

    def get_actor(self):
        """Return the actor model"""
        return self.actor

    def get_critic(self):
        """Return the critic model"""
        return self.critic