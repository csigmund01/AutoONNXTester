import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm can cause numerical differences
        self.relu1 = nn.ReLU(inplace=True)  # Inplace operation can cause issues
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Classifier layers
        self.dropout = nn.Dropout(0.5)  # Can be problematic for ONNX inference
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Auxiliary classifier for more complexity
        self.aux_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.aux_fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        # Check input dimensions - ONNX conversion issue: dynamic control flow
        if x.dim() != 4:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        # Adjust channels if grayscale wasn't provided
        if x.size(1) == 3:
            x = torch.mean(x, dim=1, keepdim=True)  # Convert RGB to grayscale
        
        # Main network branch
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        features = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Auxiliary branch - creates multiple outputs
        if self.training:
            aux = self.aux_conv(features)
            aux = aux.view(aux.size(0), -1)
            aux_output = self.aux_fc(aux)
        
        # Main branch continued
        x = self.relu3(self.bn3(self.conv3(features)))
        
        # Dynamic reshaping based on input dimensions - problematic for ONNX
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Dropout only in training mode - can cause inconsistencies
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        main_output = self.fc2(x)
        
        # Return different outputs based on mode - problematic for ONNX
        if self.training:
            return main_output, aux_output
        else:
            # Add dynamic post-processing - another ONNX issue
            probabilities = F.softmax(main_output, dim=1)
            
            # Calculate confidence score - extra operations that can cause issues
            confidence, _ = torch.max(probabilities, dim=1)
            
            # Only return highly confident predictions (dynamic filtering)
            # This is particularly problematic for ONNX
            threshold = 0.5
            mask = confidence > threshold
            return main_output, mask


if __name__ == "__main__":
    # Simple test to verify the model works
    model = MNISTClassifier()
    # Create a random batch of 4 MNIST images (1x28x28)
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # Test in eval mode
    model.eval()
    with torch.no_grad():
        outputs, mask = model(dummy_input)
        print(f"Eval mode - Output shape: {outputs.shape}, Mask shape: {mask.shape}")
    
    # Test in train mode
    model.train()
    main_out, aux_out = model(dummy_input)
    print(f"Train mode - Main output shape: {main_out.shape}, Aux output shape: {aux_out.shape}")