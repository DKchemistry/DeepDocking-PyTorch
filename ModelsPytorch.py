import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseDropoutNN(nn.Module):
    def __init__(self, input_shape, num_units, dropout_rate, output_activation):
        super(DenseDropoutNN, self).__init__()
        self.dense1 = nn.Linear(input_shape, num_units)
        self.bn1 = nn.BatchNorm1d(num_units)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(num_units, num_units)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(num_units, num_units)
        self.bn3 = nn.BatchNorm1d(num_units)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(num_units, num_units)
        self.bn4 = nn.BatchNorm1d(num_units)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(num_units, 1)
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None  # or implement other activations as needed

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.dense3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.dense4(x)))
        x = self.dropout4(x)
        x = self.output(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x