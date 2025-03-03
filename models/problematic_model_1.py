import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectableIssuesModel(nn.Module):
    """
    A PyTorch model with issues that can be detected by check_model_code
    but will still successfully run and convert to ONNX.
    These issues will be caught during static analysis and tracing.
    """
    def __init__(self):
        super(DetectableIssuesModel, self).__init__()
        # Standard layers (these are fine for ONNX)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # This implementation has patterns that check_model_code will detect
        # but will still run and convert to ONNX
        
        # Issue 1: Use of shape as tensor (will be detected in trace)
        # This will generate a warning but still work
        input_size = x.shape[2]
        if input_size >= 16:  # Will be detected by static analysis
            x = self.conv1(x)
        else:
            # Will still work with fixed input size, but check_model_code will flag it
            x = F.pad(x, (1, 1, 1, 1))
            x = F.conv2d(x, self.conv1.weight, self.conv1.bias)
            
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, 2)
        
        # Issue 2: Reshape/view operation (will be flagged but actually works with fixed shapes)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, 2)
        
        # This view operation will work with fixed batch sizes but will be detected
        # as potentially problematic
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Issue 3: Using permute (will be detected in graph analysis)
        # This will work but will be flagged by the debugger
        x = x.permute(1, 0)
        x = x.permute(1, 0)  # Restore original shape
        
        # Standard fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Issue 4: Using flatten (will be detected in graph)
        # This works but will be flagged
        x = torch.flatten(x, 1)
        
        return x

# Test function to make sure the model works
def test_model():
    model = DetectableIssuesModel()
    model.eval()
    
    # Create a sample input
    x = torch.randn(2, 3, 32, 32)
    
    # Run the model
    with torch.no_grad():
        output = model(x)
    
    print(f"Model output shape: {output.shape}")
    print("Model forward pass successful!")
    
    return model

if __name__ == "__main__":
    test_model()