import os
import base64
import torch
import numpy as np

from io import BytesIO
from PIL import Image
from skimage.color import lab2rgb
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from Network import Network  # Your model architecture
from Dataset import test_dataset, transform
from Testing import visualize_results

# Flask App Configuration
app = Flask(__name__)

# Create an upload folder (relative to current working directory)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Upload_Folder')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Same as checking and creating
app.config['UPLOAD'] = UPLOAD_FOLDER

# Detect device (prefer GPU if available, fall back to CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Model Setup
model = Network().to(device)

# Keep model file paths in a dictionary for simple lookup
MODEL_PATHS = {
    '100':  '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/colorization_net_100.pth',
    '300':  '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/colorization_net_300.pth',
    'new300': '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/new_colorization_net_300.pth',
    'new150': '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/new_colorization_net_150.pth',
}

def load_model(version: str):
    """Loads the specified model weights onto the model."""
    path = MODEL_PATHS.get(version, MODEL_PATHS['new150'])
    model.load_state_dict(torch.load(path, map_location=device))
    print(f'Model {version} loaded from {path}.')

# Helper Functions
def convert_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL Image to a Base64-encoded JPEG string."""
    buffered = BytesIO()
    pil_image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def display_images(image_sample):
    """
    Given a sample from the dataset:
      - image_sample[0] is the L-channel
      - image_sample[1] is the AB-channel
    Returns the grayscale, true RGB, and predicted RGB images (all Base64-encoded).
    """
    model.eval()
    with torch.no_grad():
        L_channel = image_sample[0].unsqueeze(0).to(device)
        ab_channels = image_sample[1].unsqueeze(0).to(device)

        predicted_ab = model(L_channel)
        results = visualize_results(L_channel, ab_channels, predicted_ab)
        gray_img, true_rgb, pred_rgb = results[:3]

        return (
            convert_to_base64(gray_img),
            convert_to_base64(true_rgb),
            convert_to_base64(pred_rgb),
        )

def colourize(image_tensor: torch.Tensor) -> str:
    """
    Colorizes a single L-channel image (uploaded by the user).
    Returns the predicted image as Base64-encoded string.
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        L_channel = image_tensor.unsqueeze(0).to(device)

        # Predict AB channels
        predicted_ab = model(L_channel)

        # Convert back to CPU and remove batch dimension
        L_channel = L_channel.squeeze().cpu()
        predicted_ab = predicted_ab.squeeze().cpu()

        # Scale L channel and combine with predicted AB
        L_channel = L_channel.unsqueeze(0) * 100
        pred_rgb = torch.cat((L_channel, predicted_ab), dim=0)
        pred_rgb = pred_rgb.permute(1, 2, 0).numpy()
        pred_rgb = lab2rgb(pred_rgb)

        # Convert to PIL and then Base64
        pred_rgb = Image.fromarray((pred_rgb * 255).astype(np.uint8))
        return convert_to_base64(pred_rgb)

# Flask Routes
@app.route("/", methods=['GET', 'POST'])
def home():
    # Retrieve possible error messages or previously uploaded images
    error_message = request.args.get('error_message', '')
    uploaded_image_b64 = ''

    if request.method == 'POST':

        # 1) Handle an uploaded file
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                selected_model = request.form.get('model', '100')
                load_model(selected_model)

                # Convert to grayscale tensor
                grayscale_img = Image.open(filepath).convert('L')
                grayscale_tensor = transform(grayscale_img)

                # Cleanup the uploaded file
                os.remove(filepath)

                # Return results to template
                return render_template(
                    'index.html',
                    gray_img=convert_to_base64(grayscale_img),
                    pred_rgb=colourize(grayscale_tensor),
                    selected_model=selected_model
                )
            else:
                # No file or an invalid file was provided
                return redirect(url_for('home', error_message="No file selected or invalid file."))

        # 2) Handle an existing test image selection
        selected_image_idx = request.form.get('image')
        if selected_image_idx:
            selected_image_idx = int(selected_image_idx)
            selected_model = request.form.get('model', '100')
            load_model(selected_model)

            # Display grayscale, true color, and predicted color
            gray_b64, true_b64, pred_b64 = display_images(test_dataset[selected_image_idx])
            return render_template(
                "index.html",
                gray_img=gray_b64,
                true_rgb=true_b64,
                pred_rgb=pred_b64,
                selected_model=selected_model,
                selected_image=selected_image_idx
            )

    # Handle a GET request
    uploaded_path = request.args.get('uploaded_image', '')
    if uploaded_path and os.path.exists(uploaded_path):
        uploaded_image_b64 = convert_to_base64(Image.open(uploaded_path))

    return render_template(
        'index.html',
        gray_img='',
        true_rgb='',
        pred_rgb='',
        selected_model='100',
        selected_image=1,
        uploaded_image=uploaded_image_b64,
        error_message=error_message
    )

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
