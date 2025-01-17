import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import lab2rgb
from torch.utils.data import DataLoader

from Network import Network
from Dataset import ColorizationDataset, test_black_path, test_color_path, transform


def plot_all_results(results, average_loss):
  num_images = len(results)
  num_cols = num_images # Horizontal Layout
  num_rows = 3 # Grayscale, True RGB, Predicted RGB

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 12))

  for i, (gray, true_rgb, pred_rgb, loss) in enumerate(results):
    # Row 0: Grayscale
    axes[0, i].imshow(gray, cmap='gray')
    axes[0, i].set_title('Grayscale')
    axes[0, i].axis('off') # Remove axis

    # Row 1: True RGB
    axes[1, i].imshow(true_rgb)
    axes[1, i].set_title('True RGB')
    axes[1, i].axis('off')

    # Row 2: Predicted RGB
    axes[2, i].imshow(pred_rgb)
    axes[2, i].set_title(f'Predicted RGB\nLoss: {loss:.4f}')
    axes[2, i].axis('off') # Remove axis

  fig.suptitle((f"Average loss across testing set:{average_loss:.4f}"))
  plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for suptitle
  plt.show()

def visualize_results(L_channel, ab_channels, predicted_ab, loss=None):
  L_channel = L_channel.squeeze().cpu() # Convert to 2 dimensions, and move it back to cpu, shape: (400, 400)
  ab_channels = ab_channels.squeeze().cpu() # shape: (2, 400, 400)
  predicted_ab = predicted_ab.squeeze().cpu() # shape: (2, 400, 400)
  
  # grayscale original image
  gray = Image.fromarray((L_channel.numpy() * 255).astype(np.uint8))

  L_channel = L_channel.unsqueeze(0) * 100 # Scale back to [0, 100] for LAB

  # true rgb image
  true_rgb = torch.cat((L_channel, ab_channels), dim=0) # Merge the channels across the channel dimension; dimension 0
  true_rgb = true_rgb.permute(1, 2, 0).numpy() # change to LAB shape from standard tensor shape, shape: (400, 400, 3)
  true_rgb = lab2rgb(true_rgb) # Values ranged [0, 1]
  true_rgb = Image.fromarray((true_rgb * 255).astype(np.uint8))

  # predicted rgb image
  pred_rgb = torch.cat((L_channel, predicted_ab), dim=0)
  pred_rgb = pred_rgb.permute(1, 2, 0).numpy()
  pred_rgb = lab2rgb(pred_rgb)
  pred_rgb = Image.fromarray((pred_rgb * 255).astype(np.uint8))
  
  return gray, true_rgb, pred_rgb, loss

# Only run if it's this program running
if __name__ == '__main__': 
  # use gpu, fallback is cpu
  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

  # Load test data
  test_dataset = ColorizationDataset(test_black_path, test_color_path, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  print('device:', device) # Ensure it runs on gpu

  # Load the trained model
  model = Network().to(device)
  model_150_path = os.path.join(os.getcwd(), 'models/new_colorization_net_150.pth')
  model_300_path = os.path.join(os.getcwd(), 'models/new_colorization_net_300.pth')

  
  # Get which model version to test from user
  while True:
    try:
      model_version = int(input("Would you like to test the 150 epoch or 300 epoch model? 150/300 \n"))
      if model_version == 150:
        model_path = model_150_path
        break
      elif model_version == 300:
        model_path = model_300_path
        break
      else:
        print("Invalid model version, please try again.")
    except ValueError:
      print('Pleaes enter a number, try again.')

  # Ensure model exists in path
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
  else:
    print(f"Model file not found at {model_path}")

  # Evaluate the model
  model.eval() # Set the model to evaluation mode

  results = []
  total_loss = 0.0
  criterion = torch.nn.MSELoss() # Define loss function; mean squared error

  start = time.time()
  print("Started testing...")

  with torch.no_grad(): # Disable model gradients for testing loop
    for batch, (L_channel, ab_channels) in enumerate(test_loader):
      # Move channels to same device
      L_channel = L_channel.to(device)
      ab_channels = ab_channels.to(device)

      # Forward pass to predict color channels
      predicted_ab = model(L_channel)

      # Compute total loss for the current batch
      loss = criterion(predicted_ab, ab_channels)
      total_loss += loss.item() # accumulate the loss
      
      # Collect results for first 5 images
      if batch < 5:
        results.append(visualize_results(L_channel, ab_channels, predicted_ab, loss.item()))

  print(f"Model Tested in:{(time.time() - start) / 60:.4f} minutes")
      
  # compute the loss for the entire testing set
  average_loss = total_loss / len(test_loader)
  plot_all_results(results, average_loss)
  print(f"Average Loss across Test Set: {average_loss:.4f}")
