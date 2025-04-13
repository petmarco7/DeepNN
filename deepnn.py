from typing import Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn

class DeepNN(nn.Module):
    """Class to create a (feed-forward) deep neural network."""

    def __init__(self, 
                input_dim: int,
                hidden_dims: Tuple[int, ...],
                output_dim: int,
                dropout: float = 0.2,
                activation=nn.ReLU
                ):
        """
        Initialize a deep neural network.

        Args:
            input_dim: Number of features in the input.
            hidden_dims: Tuple of integers representing the number of neurons in each layer.
            output_dim: Number of classes in the output.
            dropout: Dropout rate to use in hidden layers.
            activation: Activation function to use in hidden layers.
        """
        super(DeepNN, self).__init__()

        self.drop_out = nn.Dropout(dropout)
        self.activation = activation()

        # Define the layers of the neural network
        dimensions = [input_dim] + list(self._validate_hidden_dims(layers=hidden_dims)) + [output_dim]
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            self.norm_layers.append(nn.LayerNorm(dimensions[i + 1]))

        self._init_weights()

    def _validate_hidden_dims(self, layers: Tuple[int, ...]) -> Tuple[int, ...]:
        """Validate the layers tuple.
        
        Args:
            layers: Tuple of integers representing the number of neurons in each layer.
            
        Returns:
            Tuple of integers representing the number of neurons in each layer.
        """
        if not all(isinstance(n, int) for n in layers):
            raise ValueError("All elements in layers must be integers.")
        
        return layers
    
    def _init_weights(self) -> None:
        """Initialize the weights of the neural network."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        
        for norm_layer in self.norm_layers:
            norm_layer.weight.data.fill_(1)
            norm_layer.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = self.drop_out(x)
        return self.layers[-1](x)
    
    def save(self, 
             build_path: Union[str, Path] = "./build") -> None:
        """Save the model to disk.
        
        Args:
            build_path: The path to save the model.
        """
        # Convert the path to a Path object
        build_path = Path(build_path).resolve()

        # Make sure the directory exists
        build_path.mkdir(parents=True, exist_ok=True)
        build_path = build_path / "model.pth"

        # Save the model
        torch.save(self.state_dict(), build_path)

    def load(self, 
             build_path: Union[str, Path] = "./build") -> None:
        """Load the model from disk.
        
        Args:
            build_path: The path to load the model from.
        """
        # Convert the path to a Path object
        build_path = Path(build_path).resolve() / "model.pth"

        # Load the model
        self.load_state_dict(torch.load(build_path))

    def __repr__(self) -> str:
        """String representation of the model."""

        str_out = f"DeepNN(input_dim={self.layers[0].in_features}, " 
        
        for i, layer in enumerate(self.layers):
            str_out += f"layer_{i}={layer.out_features}, "

        str_out += f"output_dim={self.layers[-1].out_features})"
        return str_out
