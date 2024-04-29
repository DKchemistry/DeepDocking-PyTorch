import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchRefactoredModel(nn.Module):
    """
    Model based on my understanding that this model was otherwise hardcoded into the version of Deep Docking I have used previously.

    """

    def __init__(self, input_shape, hyperparameters):
        super(PytorchRefactoredModel, self).__init__()
        self.layers = nn.ModuleList()
        units = hyperparameters["num_units"]
        dropout_rate = hyperparameters["dropout_rate"]

        for i, layer_type in enumerate(hyperparameters["bin_array"]):
            if layer_type == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(units if i > 0 else input_shape, units),
                        nn.BatchNorm1d(units),
                        nn.ReLU(),
                    )
                )
            else:
                self.layers.append(nn.Dropout(dropout_rate))

        self.output = nn.Linear(units, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # returning logits directly b/c of loss fn 
        return self.output(x)  

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
