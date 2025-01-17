import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import rgb2lab
from PIL import Image

class ColorizationDataset(Dataset):
  def __init__(self, black_dir, color_dir, transform=None):
    self.black_paths = [os.path.join(black_dir, f) for f in os.listdir(black_dir)]
    self.color_paths = [os.path.join(color_dir, f) for f in os.listdir(color_dir)]
  
    self.transform = transform

  def __len__(self):
    return len(self.black_paths) # We know that there are equal # of images in both

  def __getitem__(self, index):
    black_img_path = self.black_paths[index]
    color_img_path = self.color_paths[index]

    # Load images
    black_img = Image.open(black_img_path).convert('L') # Grayscale
    color_img = Image.open(color_img_path).convert('RGB') # Color

    # Convert color image to LAB colour space
    color_img_lab = rgb2lab(np.array(color_img))
    ab_channels = color_img_lab[:, :, 1:]  # Color channels, model should try to generate these

    # Apply data transformations if specified; transforming pil image to tensor
    if self.transform:
        black_img = self.transform(black_img)
        ab_channels = self.transform(ab_channels)

    return black_img, torch.tensor(ab_channels, dtype=torch.float32) # Returns L_channel, ab_channels as tensors

# Set up data transformations
transform = transforms.Compose([transforms.ToTensor()]) # Convert PIL Images to Tensor

# Define dataset paths, repeated from previous "testing" code cell
path = os.path.join(os.getcwd(),"data") # Dataset directories

# Define paths to dataset
test_black_path = os.path.join(path, "test_black")
test_color_path = os.path.join(path, "test_color")
train_black_path = os.path.join(path, "train_black")
train_color_path = os.path.join(path, "train_color")

# Create dataset objects
train_dataset = ColorizationDataset(train_black_path, train_color_path, transform=transform)
test_dataset = ColorizationDataset(test_black_path, test_color_path, transform=transform)
