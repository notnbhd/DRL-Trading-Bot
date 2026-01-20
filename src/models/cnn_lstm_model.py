import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam

class CNNLSTM:
    """CNN-LSTM model for feature extraction and time series forecasting based on the paper architecture"""
    def __init__(
        self, 
        input_shape, 
        action_space, 
        learning_rate=0.00025
    ):
        """
        Initialize the CNN-LSTM model
        
        Parameters:
        -----------
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
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()
        
    def _build_actor_model(self):
        """Build the actor model using CNN-LSTM architecture"""
        # Input layer (output shape = 100, 4)
        inputs = Input(shape=self.input_shape)
        
        # Feature Learning section
        # Conv1D layer (output shape = 100, 32)
        x = Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='actor_conv_1'
        )(inputs)
        
        # MaxPooling1D layer (output shape = 50, 32)
        x = MaxPooling1D(pool_size=2, name='actor_pool_1')(x)
        
        # Sequence Learning section
        # LSTM layers (output shape = None, 32)
        x = LSTM(
            units=32,
            return_sequences=False,
            name='actor_lstm_1'
        )(x)

       
        # Dense layer (output shape = None, 32)
        x = Dense(
            units=32,
            activation='relu',
            name='actor_dense_1'
        )(x)
        
        # Output Dense layer (output shape = None, 3)
        outputs = Dense(
            units=self.action_space,
            activation='softmax',
            name='actor_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='actor')
        
        return model
    
    def _build_critic_model(self):
        """Build the critic model"""
        # Input layer (output shape = 100, 4)
        inputs = Input(shape=self.input_shape)
        
        # Feature Learning section
        # Conv1D layer (output shape = 100, 32)
        x = Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='critic_conv_1'
        )(inputs)
        
        # MaxPooling1D layer (output shape = 50, 32)
        x = MaxPooling1D(pool_size=2, name='critic_pool_1')(x)
        
        # Sequence Learning section
        # LSTM layer (output shape = None, 32)
        x = LSTM(
            units=32,
            return_sequences=False,
            name='critic_lstm_1'
        )(x)
        
        # Dense layer (output shape = None, 32)
        x = Dense(
            units=32,
            activation='relu',
            name='critic_dense_1'
        )(x)
        
        # Output Dense layer (output shape = None, 1)
        outputs = Dense(
            units=1,
            activation=None,
            name='critic_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='critic')
        
        return model
    
    def get_actor(self):
        """Return the actor model"""
        return self.actor
    
    def get_critic(self):
        """Return the critic model"""
        return self.critic 