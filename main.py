import torch
import os
from Network import Network  # Import network structure
from Dataset import test_dataset, transform
from Testing import visualize_results
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
from skimage.color import lab2rgb
import numpy as np

# Create application instance
app = Flask(__name__)
UPLOAD_FOLDER = '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/Upload_Folder'
app.config['UPLOAD'] = UPLOAD_FOLDER

# Set program to run on GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize the model and load paths
model = Network().to(device)
model_100_path = '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/colorization_net_100.pth'
model_300_path = '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/colorization_net_300.pth'
new_300_model_path = '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/new_colorization_net_300.pth'
new_150_model_path = '/Users/saiamartya/Desktop/PythonPrograms/Image-Colourizer/models/new_colorization_net_150.pth'

# Function to convert a PIL image to base64
def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format='JPEG')  # Choose appropriate format
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Function to process and display images
def display_images(image):
    model.eval()
    with torch.no_grad():
        # Move channels to GPU
        L_channel = image[0].to(device)
        ab_channels = image[1].to(device)

        # Add batch dimension to make it 4D (1, channels, height, width)
        L_channel = L_channel.unsqueeze(0)
        
        # Pass L_channel to the model
        predicted_ab = model(L_channel)
        outputs = visualize_results(L_channel, ab_channels, predicted_ab)  # Convert tensors to PIL images
        
        gray_img, true_rgb, pred_rgb = outputs[:3]
        
        # Convert images to base64
        gray_img = convert_to_base64(gray_img)
        true_rgb = convert_to_base64(true_rgb)
        pred_rgb = convert_to_base64(pred_rgb)

        return gray_img, true_rgb, pred_rgb

def load_model(model_version):
# Load weights of the selected model
    if model_version == '100':
        model.load_state_dict(torch.load(model_100_path, map_location=device))
    elif model_version == '300':
        model.load_state_dict(torch.load(model_300_path, map_location=device))
    elif model_version == 'new300':
        model.load_state_dict(torch.load(new_300_model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(new_150_model_path, map_location=device))
    print(f'Model {model_version} loaded.')

# Main flask route
@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file:  # If file exists and has a valid filename
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                selected_model = request.form.get('model', '100')
                load_model(selected_model) 

                # Convert to a black image tensor
                black_img = Image.open(filepath).convert('L')
                black_image = transform(black_img)

                os.remove(filepath)

                # Colourize and display results
                return render_template(
                    'index.html', 
                    gray_img=convert_to_base64(black_img),
                    pred_rgb=colourize(black_image, selected_model),
                    selected_model=selected_model
                )
            
            else:
                # Return an error message if no file was selected
                return redirect(url_for('home', error_message="No file selected or invalid file."))                           
        
        # If an image is chosen instead
        image = request.form.get('image')
        if image:
            selected_image = int(image)
            selected_model = request.form.get('model', '100')  # Default to '100' if not provided
            load_model(selected_model)

            return colourize(test_dataset[selected_image], selected_model, selected_image)

    # Default response for GET request
    uploaded_image = request.args.get('uploaded_image', '')  # Get uploaded image if exists
    if uploaded_image:
        uploaded_image = Image.open(uploaded_image)
        uploaded_image = convert_to_base64(uploaded_image)

    error_message = request.args.get('error_message', '')  # Get error message if exists
    return render_template(
        'index.html',
        gray_img='',
        true_rgb='',
        pred_rgb='',
        selected_model=100,
        selected_image=1,
        uploaded_image=uploaded_image,
        error_message=error_message
    )

# Wrapper function for colorizing an image
def colourize(image, selected_model, selected_image=None):
    
    # If an image was chosen:
    if selected_image is not None:
        gray_img, true_rgb, pred_rgb = display_images(image)
        return render_template(
            "index.html",
            gray_img=gray_img,
            true_rgb=true_rgb,
            pred_rgb=pred_rgb,
            selected_model=selected_model,
            selected_image=selected_image
        )
    
    # If an image was uploaded
    else:
        # Pass image through model
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            image = image.unsqueeze(0) # Add batch dimension to make it 4D (1, channels, height, width)
            predicted_ab = model(image)

            image = image.squeeze().cpu() # Convert to 2d L channel from 4d
            predicted_ab = predicted_ab.squeeze().cpu() # shape: (2, 400, 400)
            image = image.unsqueeze(0) * 100 # Scale to be compatible with predicted_ab

            # predicted rgb image
            pred_rgb = torch.cat((image, predicted_ab), dim=0)
            pred_rgb = pred_rgb.permute(1, 2, 0).numpy()
            pred_rgb = lab2rgb(pred_rgb)
            pred_rgb = Image.fromarray((pred_rgb * 255).astype(np.uint8))

            pred_rgb = convert_to_base64(pred_rgb)
            return pred_rgb

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
