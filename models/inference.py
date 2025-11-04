import torch
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import timm
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load Pretrained Models from saved files
def load_model(model_name, model_dir='./saved_models'):
    """
    Load models from saved state dictionaries
    Much faster than downloading every time!
    """
    model_path = os.path.join(model_dir, f'{model_name}.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please run 'train_and_save_models.py' first to download and save the models."
        )
    
    if model_name == "xception":
        model = timm.create_model('xception', pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    return model

# Load ImageNet class labels for visualization
def load_imagenet_labels():
    label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    import requests
    import json
    labels = requests.get(label_url).json()
    return {int(key): value[1] for key, value in labels.items()}

# Define GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.gradients = None
        self.activation = None

        def save_gradients(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def save_activation(module, input, output):
            self.activation = output

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradients)

    def generate_heatmap(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        target_score = output[:, target_class]

        # Backward pass
        target_score.backward()

        # Compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        heatmap = torch.sum(self.activation * pooled_gradients.view(1, -1, 1, 1), dim=1).squeeze()

        # Normalize and convert to numpy
        heatmap = F.relu(heatmap).cpu().detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

# Preprocess Input Image
def preprocess_image(image_path, model_name):
    if model_name == "xception":
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Xception requires 299x299 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet and VGG require 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# Visualize Heatmap and Display Class Name
def overlay_heatmap(image_path, heatmap, class_name, model_name, output_folder):
    from pathlib import Path

    original_image = Image.open(image_path).convert("RGB")
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(original_image.size)
    heatmap = np.array(heatmap)

    # Overlay heatmap
    plt.figure(figsize=(8, 8))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Original Image")

    # Plot heatmap overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Heatmap overlay
    plt.axis('off')
    plt.title(f"Heatmap Overlay\nPredicted Class: {class_name}")

    # Adjust layout to remove spacing
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Extract index from input image filename
    input_filename = Path(image_path).stem  # Get the filename without extension
    try:
        # Extract numeric part from the filename
        index = int(input_filename.split('_')[-1])
    except ValueError:
        index = 0  # Default to 0 if filename does not contain a number

    output_path = os.path.join(output_folder, f"{model_name}_output_{index}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

    print(f"  Saved output to: {output_path}")
    
    # Display the image (comment out if running without display)
    # plt.show()
    plt.close()

# Example Usage for the 3 Models
def grad_cam_for_models(image_path, output_folder):
    models_to_use = ["xception", "resnet50", "vgg19"]
    imagenet_labels = load_imagenet_labels()  # Load ImageNet class labels

    for model_name in models_to_use:
        print(f"\nProcessing {model_name}...")

        # Load the model from saved file (FAST!)
        model = load_model(model_name)
        model.eval()

        # Select the appropriate target layer
        if model_name == "xception":
            target_layer_name = "conv4.pointwise"
        elif model_name == "resnet50":
            target_layer_name = "layer4.2.conv3"
        elif model_name == "vgg19":
            target_layer_name = "features.36" 
            
        # Initialize GradCAM for the selected model
        cam = GradCAM(model, target_layer_name)

        # Preprocess the input image
        input_tensor = preprocess_image(image_path, model_name)

        # Get the predicted class
        output = model(input_tensor)
        target_class = torch.argmax(output).item()
        class_name = imagenet_labels.get(target_class, "Unknown Class")

        print(f"  Predicted class: {class_name}")

        # Generate the heatmap using GradCAM
        heatmap = cam.generate_heatmap(input_tensor, target_class)

        # Visualize the heatmap with the class label and save it
        overlay_heatmap(image_path, heatmap, class_name, model_name, output_folder)

def main():
    from datetime import datetime
    
    # Check if saved models exist
    if not os.path.exists('./saved_models'):
        print("="*60)
        print("ERROR: Saved models not found!")
        print("="*60)
        print("\nPlease run the following command first:")
        print("  python train_and_save_models.py")
        print("\nThis will download and save the models locally.")
        print("After that, you can run this script for fast inference!")
        print("="*60)
        return
    
    # Create new folder with timestamp for this analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f'../Images/analysis_{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Created new analysis folder: {output_folder}")
    print(f"{'='*60}")
    
    # Example Image Paths
    image_paths = [
        "../Images/input_image_1.jpg",  # snow leopard
        "../Images/input_image_2.jpg",  # Monument Pillar
        "../Images/input_image_3.jpg",  # War Plane (SR 71)
        "../Images/input_image_4.jpg"   # Portrait Photo
    ]
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"Processing image: {image_path}")
            print(f"{'='*60}")
            grad_cam_for_models(image_path, output_folder)
        else:
            print(f"Warning: Image not found at {image_path}")
    
    print(f"\n{'='*60}")
    print("All images processed successfully!")
    print(f"Output saved in: {output_folder}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
