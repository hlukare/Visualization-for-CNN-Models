import torch
import timm
from torchvision import models
import os

def download_and_save_models(save_dir='./saved_models'):
    """
    Download pretrained models and save them locally
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Downloading and Saving Pretrained Models")
    print("="*60)
    
    # Download and save Xception
    print("\n1. Downloading Xception model...")
    xception_model = timm.create_model('xception', pretrained=True)
    xception_path = os.path.join(save_dir, 'xception.pth')
    torch.save(xception_model.state_dict(), xception_path)
    print(f"   ✓ Xception saved to: {xception_path}")
    
    # Download and save ResNet50
    print("\n2. Downloading ResNet50 model...")
    resnet_model = models.resnet50(pretrained=True)
    resnet_path = os.path.join(save_dir, 'resnet50.pth')
    torch.save(resnet_model.state_dict(), resnet_path)
    print(f"   ✓ ResNet50 saved to: {resnet_path}")
    
    # Download and save VGG19
    print("\n3. Downloading VGG19 model...")
    vgg_model = models.vgg19(pretrained=True)
    vgg_path = os.path.join(save_dir, 'vgg19.pth')
    torch.save(vgg_model.state_dict(), vgg_path)
    print(f"   ✓ VGG19 saved to: {vgg_path}")
    
    print("\n" + "="*60)
    print("All models downloaded and saved successfully!")
    print(f"Models saved in: {os.path.abspath(save_dir)}")
    print("="*60)
    
    # Print model sizes
    print("\nModel file sizes:")
    for model_name in ['xception.pth', 'resnet50.pth', 'vgg19.pth']:
        path = os.path.join(save_dir, model_name)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {model_name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    download_and_save_models()
