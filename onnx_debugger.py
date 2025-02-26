import torch
import numpy as np
import onnx
import onnxruntime as ort
import os
import io
from contextlib import redirect_stdout, redirect_stderr

def check_onnx_conversion(model, input_shape=(3, 32, 32), batch_size=1, tolerance=1e-5):
    """
    Test a PyTorch model for ONNX conversion issues.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
        tolerance: Tolerance for output comparison
        
    Returns:
        Dictionary with test results and issue information
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create random input tensor
    input_tensor = torch.randn((batch_size,) + input_shape)
    
    # Initialize results
    results = {
        "model_name": type(model).__name__,
        "conversion_success": False,
        "output_match": False,
        "issues": []
    }
    
    # Step 1: Convert model to ONNX
    onnx_path = "temp_model.onnx"
    try:
        torch.onnx.export(
            model,
            input_tensor,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the model
        onnx.checker.check_model(onnx.load(onnx_path))
        results["conversion_success"] = True
        
    except Exception as e:
        error_msg = str(e)
        results["conversion_success"] = False
        results["issues"].append({
            "type": "conversion_error",
            "message": f"Conversion failed: {error_msg}"
        })
# ADD MORE PROBLEMATIC LAYERS HERE
        # Check for common error patterns
        error_patterns = {
            "No conversion for operator": "Model uses unsupported operations",
            "Sizes of tensors must match": "Dimension mismatch in operations",
            "dynamic": "Dynamic shape issue",
            "view": "Issue with reshape/view operation"
        }
        
        for pattern, explanation in error_patterns.items():
            if pattern in error_msg:
                results["issues"].append({
                    "message": explanation,
                    "suggestion": "Check the model for this specific issue"
                })
                break
                
        return results
    
    # Step 2: Run both PyTorch and ONNX inference
    try:
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(input_tensor)
        
        # Take first output if model returns multiple    
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]  
            
        # ONNX inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_output = session.run([output_name], {input_name: input_tensor.numpy()})[0]
        
        # Compare outputs
        pytorch_np = pytorch_output.detach().numpy()
        
        # Check shape match
        if pytorch_np.shape != onnx_output.shape:
            results["issues"].append({
                "message": f"Output shape mismatch: PyTorch {pytorch_np.shape} vs ONNX {onnx_output.shape}"
            })
            return results
        
        # Calculate differences
        max_diff = float(np.max(np.abs(pytorch_np - onnx_output)))
        results["max_difference"] = max_diff
        results["output_match"] = max_diff < tolerance
        
        if not results["output_match"]:
            # Find problematic layers
            for name, module in model.named_modules():
# ADD MORE POTENTIALLY PROBLEMATIC LAYERS HERE
                if name and (
                    (isinstance(module, torch.nn.ReLU) and getattr(module, 'inplace', False)) or
                    isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
                ):
                    results["issues"].append({
                        "message": f"Layer '{name}' may cause numerical differences",
                        "suggestion": "Check for inplace operations or BatchNorm layers"
                    })
                    
            if not results["issues"]:
                results["issues"].append({
                    "message": f"Outputs differ by {max_diff:.6f}",
                    "suggestion": "Numerical precision differences between PyTorch and ONNX"
                })
        
    except Exception as e:
        results["issues"].append({
            "message": f"Inference error: {str(e)}"
        })
        
    finally:
        # Clean up
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
    
    return results


def check_model_code(model):
    """
    Quick analysis of model's forward method for common ONNX conversion issues.
    """
    issues = []
    
    # Check if we can access the forward method
    if not hasattr(model, 'forward') or not callable(model.forward):
        issues.append({
            "message": "Cannot access model's forward method",
            "suggestion": "Ensure the model has a properly defined forward method"
        })
        return issues
    
    try:
        import inspect
        source = inspect.getsource(model.forward)
        
        # Check for common issues (simple text-based approach)
        if "if " in source and any(x in source for x in ["x.", "input", "size", "shape"]):
            issues.append({
                "message": "Dynamic control flow detected in forward method",
                "suggestion": "Avoid using input dimensions in conditional statements"
            })
            
        if ".view(" in source or ".reshape(" in source:
            issues.append({
                "message": "Dynamic reshaping operations may cause issues",
                "suggestion": "Use fixed dimensions when possible"
            })
    except Exception as e:
        issues.append({
            "message": f"Could not inspect forward method source: {str(e)}",
            "suggestion": "Model may use a built-in or compiled forward method"
        })
    
    return issues


def print_test_results(results):
    """Print test results in a readable format"""
    print("\n" + "="*60)
    print(" ONNX CONVERSION TEST RESULTS ")
    print("="*60)
    
    print(f"\nModel: {results['model_name']}")
    print(f"Conversion Success: {results['conversion_success']}")
    print(f"Outputs Match: {results.get('output_match', False)}")
    
    if results.get('max_difference') is not None:
        print(f"Maximum Output Difference: {results['max_difference']:.6f}")
    
    if results["issues"]:
        print("\nISSUES DETECTED:")
        for i, issue in enumerate(results["issues"], 1):
            print(f"\nIssue {i}: {issue['message']}")
            if issue.get('suggestion'):
                print(f"Suggestion: {issue['suggestion']}")
    else:
        print("\nNo issues detected! Model should work well with ONNX.")
    
    print("\n" + "="*60)


def test_onnx_compatibility(model, input_shape=None, batch_size=1):
    """
    Test a PyTorch model for ONNX compatibility and print results.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
    """
    # Run conversion and inference tests
    results = check_onnx_conversion(model, input_shape, batch_size)
    
    # Add static code analysis results
    code_issues = check_model_code(model)
    results["issues"].extend(code_issues)
    
    # Print results
    print_test_results(results)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example model with known ONNX conversion issues
    class ProblemModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU(inplace=True)  # Inplace operation can cause issues
            self.pool = torch.nn.MaxPool2d(2)
            self.fc = torch.nn.Linear(16 * 16 * 16, 10)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            
            # Dynamic control flow based on input - problematic for ONNX
            if x.size(0) > 1:
                x = x * 2
            
            # Dynamic shape handling - can cause issues
            x = x.view(x.size(0), -1)
            
            return self.fc(x)
    
    # Test the model
    model = ProblemModel()
    test_onnx_compatibility(model, input_shape=(3, 32, 32))