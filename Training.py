import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Network import Network
from Dataset import train_dataset
import matplotlib.pyplot as plt
import os
import time

# Hyperparemeters
learning_rate = 0.001
epochs = 100
batch_size = 32

# Device configuration: Makes sure pytorch uses gpu if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # mps to use apple metal performance shaders; to run on mac. Otherwise 'cuda' for google colab.
print("device:", device)

# Load data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the model, loss function and optimizer
torch.manual_seed(41)
model = Network().to(device) # Save it to gpu
criterion = nn.MSELoss() # Recomended loss criterion for tasks like this
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track loss for every 200 images, to plot later
losses = []

start = time.time()
# Training loop
for epoch in range(epochs):
  model.train() # Set the pytorch model to training mode
  running_loss = 0.0
  
  for batch, (L_channel, ab_channels) in enumerate(train_loader):
    L_channel =  L_channel.to(device) # L channel input
    ab_channels = ab_channels.to(device) # ab channels target

    # Forward pass
    outputs = model(L_channel) # Pass the L channel
    loss = criterion(outputs, ab_channels) # calculate the loss (how "off" it is)

    # Backward pass: Update our paremeters
    optimizer.zero_grad() # Reset model gradients to 0
    loss.backward() # Calculate new gradients
    optimizer.step() # Update model

    running_loss += loss.item()

    # Record loss every 150 batches
    if (batch + 1) % 150 == 0:
      average_loss = running_loss / 150
      losses.append(average_loss)
      print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch+1}], Loss: {average_loss:.4f}")
      running_loss = 0.0

print(f"Model Trained in:{(time.time() - start) / 60:.4f} minutes")

# Save the trained model
os.makedirs('models', exist_ok=True)
model_path = 'models/colorization_net.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Also save the optimizer for future training
torch.save(optimizer.state_dict(), 'models/optimizer_state.pth')

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Loss per 150 batches')
plt.xlabel('Steps (x4800 Images)')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid()
plt.savefig('loss_curve.png')  # Save the plot
plt.show()
