from helper_functions.test_onnx_conversion import test_onnx_conversion
from helper_functions.static_model_analysis import static_model_analysis


def print_test_results(results):
    """Print test results in a readable format"""
    print("\n" + "="*60)
    print(" ONNX CONVERSION TEST RESULTS ")
    print("="*60)
    
    print(f"\nModel: {results['model_name']}")
    print(f"Conversion Success: {results['conversion_success']}")
    print(f"Outputs Match: {results.get('output_match', False)}")
    print(f"Maximum Output Difference: {results['max_difference']:.6f}")
    
    # Print static analysis issues
    if results.get("static_issues"):
        print("\nSTATIC ANALYSIS ISSUES:")
        for i, issue in enumerate(results["static_issues"], 1):
            print(f"\nIssue {i}: {issue['message']}")
            if issue.get('suggestion'):
                print(f"Suggestion: {issue['suggestion']}")
    
    # Print conversion issues
    if results.get("issues"):
        print("\nCONVERSION ISSUES:")
        for i, issue in enumerate(results["issues"], 1):
            print(f"\nIssue {i}: {issue['message']}")
            if issue.get('suggestion'):
                print(f"Suggestion: {issue['suggestion']}")
    
    if not results.get("static_issues") and not results.get("issues"):
        print("\nNo issues detected! Model should work well with ONNX.")
    
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
    
    # Then run conversion and inference tests
    results = test_onnx_conversion(model, input_shape, batch_size)
    
    # Add static issues separately
    results["static_issues"] = static_issues
    
    # Print results
    print_test_results(results)
    
    return results


# # Example MNIST usage
# if __name__ == "__main__":
#     # Import the MNIST model from separate file
#     from models.mnist_model import MNISTClassifier
    
#     # Create and test the model
#     model = MNISTClassifier()
    
#     # MNIST images are 1x28x28
#     test_onnx_compatibility(model, input_shape=(1, 28, 28))

# Example problematic model 1 usage
if __name__ == "__main__":
    # Import the MNIST model from separate file
    from models.problematic_model_1 import DetectableIssuesModel
    
    # Create and test the model
    model = DetectableIssuesModel()
    
    # MNIST images are 1x28x28
    onnx_debugger(model, input_shape=(3, 32, 32))
