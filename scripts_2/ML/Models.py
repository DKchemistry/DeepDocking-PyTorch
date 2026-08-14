import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchRefactoredModel(nn.Module):
    def __init__(self, input_shape, hyperparameters):
        super(PytorchRefactoredModel, self).__init__()
        self.layers = nn.ModuleList()
        units = hyperparameters["num_units"]
        dropout_rate = hyperparameters["dropout_rate"]

        # Debug print statements to check values
        print(
            f"Initializing model with input_shape: {input_shape}, units: {units}, dropout_rate: {dropout_rate}"
        )

        for i, layer_type in enumerate(hyperparameters["bin_array"]):
            if layer_type == 0:
                print(
                    f"Adding Linear Layer with input_dim: {units if i > 0 else input_shape}, output_dim: {units}"
                )
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(units if i > 0 else input_shape, units),
                        nn.BatchNorm1d(units),
                        nn.ReLU(),
                    )
                )
            else:
                print(f"Adding Dropout Layer with rate: {dropout_rate}")
                self.layers.append(nn.Dropout(dropout_rate))

        self.output = nn.Linear(units, 1)
        print(f"Output layer: {units} -> 1")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # returning logits directly b/c of loss fn
        return self.output(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @staticmethod
    # this is a static method b/c it doesn't need `self` but it is related to handling this class
    def load(path, input_shape, hyperparameters):
        model = PytorchRefactoredModel(input_shape, hyperparameters)  # init model
        model.load_state_dict(
            torch.load(path)
        )  # load weights into init model based on path
        return model  # return model with loaded weights
