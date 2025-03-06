from helper_functions.test_onnx_conversion import test_onnx_conversion
from helper_functions.static_model_analysis import static_model_analysis
from helper_functions.compare_models import compare_models

import torchvision.models as models

def print_analysis_results(results, label="ANALYSIS RESULTS"):
    """
    Print a single analysis results dictionary in a readable format.
    
    Args:
        results: Dictionary containing analysis results
        label: Label to identify what kind of results are being printed
    """
    print("\n" + "="*60)
    print(f" {label} ")
    print("="*60)
    
    # Print basic model information if available
    # if "model_name" in results:
    #     print(f"\nModel: {results['model_name']}")
    
    # Print conversion status if available
    if "conversion_success" in results:
        print(f"Conversion Success: {results['conversion_success']}")
    
    # Print output match status if available
    if "output_match" in results:
        print(f"Outputs Match: {results['output_match']}")
    
    # Print difference metrics if available
    if "max_difference" in results:
        print(f"Maximum Output Difference: {results['max_difference']:.6f}")
    
    # Print issues if available
    if "issues" in results and results["issues"]:
        print("\nISSUES:")
        for i, issue in enumerate(results["issues"], 1):
            print(f"\nIssue {i}: {issue.get('message', 'Unknown issue')}")
            if issue.get('suggestion'):
                print(f"Suggestion: {issue['suggestion']}")
    else:
        print("\nNo issues detected!")
    
    print("\n" + "="*60)


def onnx_debugger(model, input_shape=None, batch_size=1):
    """
    Test a PyTorch model for ONNX compatibility and print results.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
    """
    # Run static code analysis first
    static_issues = static_model_analysis(model, input_shape, batch_size)
    print_analysis_results(static_issues, "STATIC ANALYSIS RESULTS")
    # Then run conversion and inference tests
    conversion_issues = test_onnx_conversion(model, input_shape, batch_size)
    print(conversion_issues)
    print_analysis_results(conversion_issues, "CONVERSION ANALYSIS RESULTS")
    if not conversion_issues["conversion_success"]:
        return
    
    compare_models(model, conversion_issues["onnx_path"])



# Example MNIST usage
# if __name__ == "__main__":
#     # Import the MNIST model from separate file
#     from models.mnist_model import MNISTClassifier
    
#     # Create and test the model
#     model = MNISTClassifier()
    
#     # MNIST images are 1x28x28
#     onnx_debugger(model, input_shape=(1, 28, 28))

if __name__ == "__main__":

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    onnx_debugger(model, input_shape=(3, 224, 224))