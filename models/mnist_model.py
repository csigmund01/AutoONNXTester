import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def test_model():
    # Create a model instance
    model = MNISTClassifier()
    
    # Create a random batch of MNIST images
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # Test in eval mode
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        # Check that outputs have reasonable values
        print(f"Output shape: {outputs.shape}")
        print(f"Output values:\n{outputs}")
        
        # Check if predictions are distributed (not stuck on one class)
        predictions = outputs.argmax(dim=1)
        print(f"Predictions: {predictions}")
        
        # Check if outputs have reasonable magnitudes
        print(f"Min output: {outputs.min().item()}, Max output: {outputs.max().item()}")
        print(f"Mean absolute value: {outputs.abs().mean().item()}")


if __name__ == "__main__":
    # Test the model
    test_model()