import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime as ort
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from torchvision.models.resnet import ResNet50_Weights
import random

def compare_models(
    pytorch_model: nn.Module,
    processor: any, 
    onnx_model_path: str,
    num_matching_samples: int = 100,
    num_differing_samples: int = 100,
    batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compare PyTorch and ONNX models using Food101 to find images where:
    1) Both models predict the same class
    2) Models predict different classes
    
    Args:
        pytorch_model: The PyTorch model
        processor: Image processor for the model
        onnx_model_path: Path to the converted ONNX model
        num_matching_samples: Number of matching prediction samples to collect
        num_differing_samples: Number of differing prediction samples to collect
        batch_size: Batch size for processing
        device: Device to run PyTorch model on
        
    Returns:
        Tuple of two lists:
        - List of samples where both models predict the same class
        - List of samples where models predict different classes
    """
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)

    # Set up PyTorch model
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    # Set up ONNX runtime session with GPU provider if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    
    # Use the processor's transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((processor.size["height"], processor.size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # Load Food101 dataset
    #test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)


    # dataset_size = len(full_dataset)
    # subset_size = int(dataset_size * 0.25)  # 5% of the dataset
    # indices = random.sample(range(dataset_size), subset_size)
    # test_dataset = torch.utils.data.Subset(full_dataset, indices)
    # print(f'Created subset with {subset_size} samples (5% of {dataset_size})')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create a dictionary mapping indices to labels
    class_labels = pytorch_model.config.id2label if hasattr(pytorch_model, 'config') else test_dataset.classes
    id_to_label = {int(i): label for i, label in class_labels.items()} if isinstance(class_labels, dict) else {i: label for i, label in enumerate(class_labels)}
    
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
            # Handle different output types from the model
            if hasattr(pytorch_outputs, 'logits'):
                pytorch_outputs = pytorch_outputs.logits
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
                'true_class': id_to_label.get(true_label, f"class_{true_label}"),
                'pytorch_pred_label': pytorch_pred,
                'pytorch_pred': id_to_label.get(pytorch_pred, f"class_{pytorch_pred}"),
                'onnx_pred_label': onnx_pred,
                'onnx_pred': id_to_label.get(onnx_pred, f"class_{onnx_pred}")
            }
            
            # Check if predictions match
            if pytorch_pred == onnx_pred:
                if len(matching_samples) < num_matching_samples:
                    matching_samples.append(sample_info)
            else:
                if len(differing_samples) < num_differing_samples:
                    differing_samples.append(sample_info)
    save_results_to_csv(matching_samples, differing_samples, "output_csv.csv")

    print(f"Found {len(matching_samples)} matching samples and {len(differing_samples)} differing samples")
    return matching_samples, differing_samples

def save_results_to_csv(matching_samples, differing_samples, output_csv_path):
    """
    Save the comparison results to a CSV file.
    
    Args:
        matching_samples: List of samples where both models predict the same class
        differing_samples: List of samples where models predict different classes
        output_csv_path: Path to save the CSV file
    """
    # Define the columns we want to save (exclude image tensor)
    csv_columns = [
        'sample_type',  # New column to differentiate matching vs differing
        'sample_index',  # To keep track of original index in the respective list
        'true_label',
        'true_class',
        'pytorch_pred_label',
        'pytorch_pred',
        'onnx_pred_label',
        'onnx_pred',
        'match_status'  # Whether predictions match or not
    ]
    
    print(f"Saving results to {output_csv_path}...")
    import csv
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            
            # Write matching samples
            for i, sample in enumerate(matching_samples):
                row = {
                    'sample_type': 'matching',
                    'sample_index': i,
                    'true_label': sample['true_label'],
                    'true_class': sample['true_class'],
                    'pytorch_pred_label': sample['pytorch_pred_label'],
                    'pytorch_pred': sample['pytorch_pred'],
                    'onnx_pred_label': sample['onnx_pred_label'],
                    'onnx_pred': sample['onnx_pred'],
                    'match_status': 'match'
                }
                writer.writerow(row)
            
            # Write differing samples
            for i, sample in enumerate(differing_samples):
                row = {
                    'sample_type': 'differing',
                    'sample_index': i,
                    'true_label': sample['true_label'],
                    'true_class': sample['true_class'],
                    'pytorch_pred_label': sample['pytorch_pred_label'],
                    'pytorch_pred': sample['pytorch_pred'],
                    'onnx_pred_label': sample['onnx_pred_label'],
                    'onnx_pred': sample['onnx_pred'],
                    'match_status': 'mismatch'
                }
                writer.writerow(row)
                
        print(f"Successfully saved {len(matching_samples) + len(differing_samples)} samples to {output_csv_path}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def display_differing_predictions(differing_samples, n=4, save_path="model_differences.png"):
    """Display a grid of images where PyTorch and ONNX models disagree."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Handle the case when there are no differing samples
    if not differing_samples:
        print("No differing samples found to display.")
        return
    
    # Handle the case when there are few samples
    n = min(n, len(differing_samples))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3))
    
    # Make axes a list even if there's only one subplot
    if n == 1:
        axes = [axes]
    
    for i, sample in enumerate(differing_samples[:n]):
        # Get image and denormalize
        img = sample['image_tensor'].numpy().transpose(1, 2, 0)
        
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        # Denormalize the image
        img = img * std + mean
        
        # Ensure values are within valid range
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {sample['true_class']}\nPyTorch: {sample['pytorch_pred']}\nONNX: {sample['onnx_pred']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    # import torchvision.models as models
    
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # model.fc = torch.nn.Linear(model.fc.in_features, 101)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    # processor = AutoImageProcessor.from_pretrained("nateraw/food")
    # model = AutoModelForImageClassification.from_pretrained("nateraw/food")
    
    processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")
    model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
    
    onnx_model_path = "converted_model.onnx"

    matching_samples, differing_samples = compare_models(
        pytorch_model=model,
        processor=processor,
        onnx_model_path=onnx_model_path,
        batch_size=64
    )
    
    # Display images where models disagree and save to PNG
    display_differing_predictions(
        differing_samples, 
        n=4, 
        save_path="model_differences.png"
    )