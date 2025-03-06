#from helper_functions.static_model_analysis import static_model_analysis
from helper_functions.convert_to_onnx import convert_to_onnx
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

    if "conversion_success" in results:
        print(f"Conversion Success: {results['conversion_success']}")
    
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
    #static_issues = static_model_analysis(model, input_shape, batch_size)
    #print_analysis_results(static_issues, "STATIC ANALYSIS RESULTS")
    
    # Then run conversion and inference tests
    conversion_issues = convert_to_onnx(model, input_shape, batch_size)
    print(conversion_issues)
    print_analysis_results(conversion_issues, "CONVERSION ANALYSIS RESULTS")
    if not conversion_issues["conversion_success"]:
        return
    
    compare_models(model, conversion_issues["onnx_path"])


if __name__ == "__main__":
    import torchvision.models as models
    
    # Load pre-trained ViT model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    onnx_debugger(model, input_shape=(3, 224, 224))