import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime as ort
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import time

def compare_models(
    pytorch_model: nn.Module,
    onnx_model_path: str,
    num_matching_samples: int = 100,
    num_differing_samples: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compare PyTorch and ONNX models using CIFAR-10 to find images where:
    1) Both models predict the same class
    2) Models predict different classes
    
    Args:
        pytorch_model: The PyTorch model
        onnx_model_path: Path to the converted ONNX model
        num_matching_samples: Number of matching prediction samples to collect
        num_differing_samples: Number of differing prediction samples to collect
        batch_size: Batch size for processing
        device: Device to run PyTorch model on
        
    Returns:
        Tuple of two lists:
        - List of samples where both models predict the same class
        - List of samples where models predict different classes
        
        Each sample contains:
        - image_idx: Unique identifier for the image
        - image_tensor: The image tensor
        - true_label: The integer class ID
        - true_class: The human-readable class name
    """
    # Set up PyTorch model
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    # Set up ONNX runtime session with GPU provider if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    
    # ViT preprocessing for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(224),  # ViT requires 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    
    # CIFAR-10 class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    id_to_label = {i: name for i, name in enumerate(cifar10_classes)}
    
    matching_samples = []
    differing_samples = []
    
    print("Comparing PyTorch and ONNX model predictions...")
    for images, labels in tqdm(test_loader):
        # Skip if we have enough samples
        if (len(matching_samples) >= num_matching_samples and 
            len(differing_samples) >= num_differing_samples):
            break
            
        # PyTorch forward pass
        with torch.no_grad():
            pytorch_outputs = pytorch_model(images.to(device))
            pytorch_predictions = torch.argmax(pytorch_outputs, dim=1).cpu().numpy()
        
        # ONNX batch inference
        onnx_outputs = ort_session.run(None, {input_name: images.numpy()})
        onnx_predictions = np.argmax(onnx_outputs[0], axis=1)
        
        # Process each image in the batch
        for i, (image, label, pytorch_pred, onnx_pred) in enumerate(
            zip(images, labels, pytorch_predictions, onnx_predictions)
        ):
            # Skip if we have enough samples of both types
            if (len(matching_samples) >= num_matching_samples and 
                len(differing_samples) >= num_differing_samples):
                break
                
            true_label = label.item()
            
            # Create basic sample info
            sample_info = {
                'image_idx': i,
                'image_tensor': image,
                'true_label': true_label,
                'true_class': id_to_label.get(true_label, f"class_{true_label}")
            }
            
            # Check if predictions match
            if pytorch_pred == onnx_pred:
                if len(matching_samples) < num_matching_samples:
                    matching_samples.append(sample_info)
            else:
                if len(differing_samples) < num_differing_samples:
                    differing_samples.append(sample_info)
    
    print(f"Found {len(matching_samples)} matching samples and {len(differing_samples)} differing samples")
    return matching_samples, differing_samples


# Example usage:
if __name__ == "__main__":
    import torchvision.models as models
    import os
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained ViT model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()
    
    # Convert to ONNX if needed
    onnx_model_path = "vit_b_16.onnx"
    if not os.path.exists(onnx_model_path):
        print("Converting PyTorch model to ONNX...")
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_model_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=14
        )
        print(f"Model exported to {onnx_model_path}")
    
    # Compare models
    matching_samples, differing_samples = compare_models(
        pytorch_model=model,
        onnx_model_path=onnx_model_path,
        device=device
    )