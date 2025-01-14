import os

from PIL import Image
from Dataset import ColorizationDataset

path = "/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/data"

# Define paths to dataset
test_black_path = os.path.join(path, "test_black")
test_color_path = os.path.join(path, "test_color")
train_black_path = os.path.join(path, "train_black")
train_color_path = os.path.join(path, "train_color")

# Sorted paths for training and testing datasets
train_black_paths = sorted([os.path.join(train_black_path, f) for f in os.listdir(train_black_path)])
train_color_paths = sorted([os.path.join(train_color_path, f) for f in os.listdir(train_color_path)])

test_black_paths = sorted([os.path.join(test_black_path, f) for f in os.listdir(test_black_path)])
test_color_paths = sorted([os.path.join(test_color_path, f) for f in os.listdir(test_color_path)])

# Was showing .DS_store because is a metadata file in macos for view options
# Ensure there is no .DS_Store file
if os.path.basename(train_color_paths[0]) == '.DS_Store':
    os.remove(train_color_paths[0])
    train_color_paths.pop(0)
if os.path.basename(test_color_paths[0]) == '.DS_Store':
    os.remove(test_color_paths[0])
    test_color_paths.pop(0)

print("Before length of train black:", len(train_black_paths))
print("Before length of train color:", len(train_color_paths))
print("Before length of test black:", len(test_black_paths))
print("Before length of test color:", len(test_color_paths))

# Efficiently delete all black train photos that have no color copies
black_pointer, color_pointer = 0, 0

while black_pointer < len(train_black_paths):
    black_name = os.path.basename(train_black_paths[black_pointer])
    color_name = os.path.basename(train_color_paths[color_pointer])
    
    if black_name == color_name:
        black_pointer += 1
        color_pointer += 1
    elif black_name < color_name:
        # Remove unmatched black path
        os.remove(train_black_paths[black_pointer])
        train_black_paths.pop(black_pointer)

# Efficiently delete all black test photos that have no color copies
black_pointer, color_pointer = 0, 0

while black_pointer < len(test_black_paths):
    black_name = os.path.basename(test_black_paths[black_pointer])
    color_name = os.path.basename(test_color_paths[color_pointer])
    
    if black_name == color_name:
        black_pointer += 1
        color_pointer += 1
    elif black_name < color_name:
        # Remove unmatched black path
        os.remove(test_black_paths[black_pointer])
        test_black_paths.pop(black_pointer)
    else:
        color_pointer += 1  # Only move color_pointer forward when black_name > color_name

print("Length of train black:", len(train_black_paths))
print("Length of train color:", len(train_color_paths))
print("Length of test black:", len(test_black_paths))
print("Length of test color:", len(test_color_paths))
