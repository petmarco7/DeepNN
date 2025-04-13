from typing import Callable
import torch
import torch.optim as optim
import torch.utils.data as tdata
from tqdm import tqdm

from deepnn import DeepNN

class Trainer:
    """Class to train a DeepNN model"""

    def __init__(self, 
                 model: DeepNN, 
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: optim.Optimizer = optim.Adam,
                 scheduler: optim.lr_scheduler._LRScheduler | None = None,
                 ):
        """Initialize the trainer.

        Args:
            model: The model to train.
            criterion: The loss function to use.
            optimizer: The optimizer to use.
            scheduler: The learning rate scheduler to use.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, 
              train_loader: tdata.DataLoader, 
              val_loader: tdata.DataLoader | None = None,
              epochs: int = 100) -> None:
        """Train the model.
        
        Args:
            train_loader: The training data loader.
            val_loader: The validation data loader.
            epochs: The number of epochs to train for.
        """
        train_best_loss = float("inf")
        val_best_loss = float("inf")

        progress_bar = tqdm(range(epochs), desc="Training")
        
        for epoch in range(epochs):
            
            # Set the model to training mode
            self.model.train()

            # Training loop
            training_losses = []

            # Iterate over the training batches
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                training_losses.append(loss.item())

            # Calculate the average training loss
            running_loss = sum(training_losses) / len(training_losses)

            if running_loss < train_best_loss:
                train_best_loss = running_loss

                if not val_loader:
                    self.model.save()

            if val_loader:
                # Set the model to evaluation mode
                self.model.eval()
                val_losses = []

                # Iterate over the validation batches
                for i, (inputs, labels) in enumerate(val_loader):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_losses.append(loss.item())

                # Calculate the average validation loss
                running_val_loss = sum(val_losses) / len(val_losses)

                if running_val_loss < val_best_loss:
                    val_best_loss = running_val_loss
                    self.model.save()

            progress_bar.set_postfix({"training_loss": running_loss, "best_validation_loss": val_best_loss})
            progress_bar.update()

            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Training is done, load the best model
        self.model.load()
        return val_best_loss

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels for the data."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)