from deepnn import DeepNN
from trainer import Trainer

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

import matplotlib.pyplot as plt
import random

# Create a true model
torch.manual_seed(33)
random.seed(33)

def true_model(x):
    return 2 * x + 1 + 25*torch.cos( x )

# Create a dataset
a = -100
b = 100
X = torch.linspace(a, b, 1000).view(-1, 1)
y = true_model(X)

 # Create a noisy dataset
X_data = random.sample(list(X), 250)
X_data = torch.tensor(X_data).float().view(-1, 1)

# Append cos(X) to the dataset
y_data = true_model(X_data) + 10*torch.randn(X_data.shape)
X_data = torch.cat((X_data, torch.cos(X_data)), dim=1)

# Normalize the dataset
X_data_mean = X_data.mean()
X_data_std = X_data.std()
X_data = (X_data - X_data_mean) / X_data_std

def make_prediction(
        learning_rate: float,
        weight_decay: float,
        hidden_dims: tuple,
        dropout: float,
        step_size: int,
        gamma: float
):

    dataset = TensorDataset(X_data, y_data)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1000)
    val_loader = DataLoader(val_dataset, batch_size=1000)

    # Create a DeepNN model
    model = DeepNN(input_dim=2, 
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout=dropout,
                    activation=torch.nn.LeakyReLU
                )

    # Create a Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = None

    trainer = Trainer(model=model, 
                    criterion=torch.nn.MSELoss(), 
                        optimizer=optimizer,
                        scheduler=scheduler)

    # Train the model
    best_val_loss = trainer.train(train_loader, val_loader, epochs=1000)
    return trainer.model, best_val_loss

params = {'dropout': 0.2, 'gamma': 0.5, 'hidden_dims': (50, 50, 50), 'learning_rate': 0.01, 'step_size': 1000, 'weight_decay': 0.001}

best_model, best_loss = make_prediction(**params)

# Plot the dataset
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='True Model')

# Denormalize the X values
X_data = X_data * X_data_std + X_data_mean
plt.scatter(X_data[:,0], y_data, c='r', label='Noisy Data')

# Normalize the X values

X_pred = torch.cat((X, torch.cos(X)), dim=1)
X_pred = (X_pred - X_pred.mean()) / X_pred.std()

# Make predictions
best_model.eval()
y_pred = best_model(X_pred).detach().numpy()

plt.plot(X, y_pred, label='Predicted Model', c='g')

plt.xlabel('X')
plt.ylabel('y')

plt.legend()
plt.show()

# from sklearn.model_selection import ParameterGrid

# parameters = {
#     "learning_rate": [0.01, 0.001],
#     "weight_decay": [0.01, 0.001],
#     "hidden_dims": [(50, 50, 50)],
#     "dropout": [0.2],
#     "step_size": [800, 1000],
#     "gamma": [0.1, 0.2, 0.3]
# }

# grid = ParameterGrid(parameters)

# best_loss = float("inf")
# best_params = None

# for params in grid:
#     print(f"Training with parameters: {params}")
#     val_best_loss = make_prediction(**params)
#     if val_best_loss < best_loss:
#         best_loss = val_best_loss
#         best_params = params

# print(f"Best parameters: {best_params}")
# print(f"Best loss: {best_loss}")