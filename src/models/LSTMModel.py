import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers_lstm, num_layers_dense, hidden_size_multiplier=1.0, dropout=0.2, norm_type=None):
        super(LSTMModel, self).__init__()
        
        # Calculate hidden_size using the multiplier
        hidden_size = int(input_size * hidden_size_multiplier)
        
        # LSTM layers with Dropout and Normalization
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers_lstm):
            lstm_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
            
            if norm_type == 'layernorm':
                self.norm_layers.append(nn.LayerNorm(hidden_size))
            elif norm_type == 'rmsnorm':
                self.norm_layers.append(RMSNorm(hidden_size))
            else:
                self.norm_layers.append(None)
        
        # Dense layers with Dropout
        self.dense_layers = nn.ModuleList()
        dense_input_size = hidden_size
        for i in range(num_layers_dense):
            dense_output_size = output_size if i == num_layers_dense - 1 else hidden_size
            self.dense_layers.append(nn.Linear(dense_input_size, dense_output_size))
            if i < num_layers_dense - 1:
                self.dense_layers.append(nn.ReLU())
                self.dense_layers.append(nn.Dropout(dropout))
            dense_input_size = hidden_size

    def forward(self, x):
        # Passing through LSTM layers with Dropout and Normalization
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            x = self.dropout_layers[i](x)
            if self.norm_layers[i] is not None:
                x = self.norm_layers[i](x)
        
        # Taking the last time step
        x = x[:, -1, :]
        
        # Passing through Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        return x

