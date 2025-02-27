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

        # Check for common error patterns
        error_patterns = {
            # Node conversion errors - primary conversion issues
            "No conversion for operator": "Model uses unsupported operations (common node conversion error)",
            "Sizes of tensors must match": "Dimension mismatch in operations (common node conversion error)",
            "dynamic": "Dynamic shape issue (common in graph tracing)",
            "view": "Issue with reshape/view operation (frequently fails in ONNX conversion)",
            "reshape": "Issue with reshape operation (common source of incompatibility)",
            "Converting a tensor to a Python boolean": "Tensor dimension used in conditional statement (causes tracing failures)",
            "expected scalar type": "Data type mismatch during conversion (type problem)",
            "Overflow": "Numeric overflow during conversion (precision issue)",
            
            # Tensor operation issues
            "unbind": "Issue with tensor unbind operation (often problematic in ONNX)",
            "scatter": "Issue with scatter/gather operations (limited support in ONNX)",
            "indices": "Problem with index-based operations (different behavior in ONNX)",
            "broadcast": "Broadcasting dimension mismatch (incompatible broadcasting rules)",
            "symbolic shape": "Unable to determine symbolic shape (dynamic shape issue)",
            "transpose": "Issue with tensor permutation (often requires explicit reshaping)",
            "slice": "Problem with slice operations (differences in slice semantics)",
            
            # Unsupported features and operations
            "unsupported": "Unsupported operation or attribute (check ONNX operator support)",
            "LSTM": "Issue with LSTM or RNN conversion (often requires special handling)",
            "RNN": "Issue with RNN conversion (limited compatibility)",
            "GRU": "Issue with GRU conversion (may need simplified architecture)",
            "inplace": "Inplace operation causing conversion issue (not supported in ONNX graph)",
            "getitem": "Problem with tensor indexing (different semantics in ONNX)",
            "matmul": "Matrix multiplication issue (dimension or type mismatch)",
            "NaN": "Not-a-number values detected (numerical stability issue)",
            
            # Tracing and graph conversion issues
            "torch.jit.trace": "Tracing failed to capture dynamic operations (try script mode instead)",
            "must be constant": "Expected constant value but found dynamic input (graph construction issue)",
            "Incompatible type promotion": "Type conversion incompatibility between operators (common in mixed precision models)",
            "overload resolution": "Ambiguous operator overload resolution (not properly traced)",
            
            # Specific operator issues (found in paper to cause problems)
            "topk": "Issue with topk operation (dynamic k values are problematic)",
            "softmax": "Issue with softmax operation (dimension handling issues)",
            "pooling": "Issue with pooling operation (padding or dimension mismatch)",
            "padding": "Issue with padding operation (inconsistent padding rules)",
            "Not implemented": "ONNX operator not implemented for this version (check opset compatibility)",
            "only supports": "Unsupported operator parameter or configuration (limited ONNX support)",
            "expected a single value": "Expected scalar but received tensor (type mismatch)",
            "Opset version": "Operator not supported in selected ONNX opset version (try updating opset)",
            "name already exists": "Duplicate node names in ONNX graph (naming conflict in conversion)",
            "Type Error": "Type mismatch between PyTorch and ONNX operators (conversion incompatibility)",
            "inputs are non-homogeneous": "Different input types for operation (type uniformity required)",
            
            # Operator sequence issues (identified in research)
            "sequential module": "Issues in sequential layer conversion (check layer sequence compatibility)",
            "graph optimization": "Graph optimization failed (complex operator sequence issue)",
            "operator fusion": "Operator fusion issue (problematic operator combinations)",
            "custom layer": "Custom layer conversion failed (not supported in ONNX)"
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


def check_model_code(model, input_shape=(3, 32, 32), batch_size=1):
    """
    Analysis of model's forward method for common ONNX conversion issues
    using both static inspection and dynamic tracing approaches.
    
    Args:
        model: PyTorch model to analyze
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
    """
    issues = []
    
    # Part 1: Static analysis of source code (if available)
    import inspect
    
    # Check for forward method existence
    if not hasattr(model, 'forward') or not callable(model.forward):
        issues.append({
            "message": "Cannot access model's forward method",
            "suggestion": "Ensure the model has a properly defined forward method"
        })
    else:
        # Try to inspect the source code
        try:
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
    
    # Part 2: Dynamic analysis using PyTorch's JIT tracing
    try:
        # Create an input tensor for tracing the model
        input_tensor = torch.randn((batch_size,) + input_shape)
        
        # Set model to evaluation mode before tracing
        model.eval()
        
        # Capture stdout/stderr during tracing to analyze warnings
        output_capture = io.StringIO()
        with redirect_stdout(output_capture), redirect_stderr(output_capture):
            # Create a traced version of the model
            traced_model = torch.jit.trace(model, input_tensor)
        
        # Get the trace output and graph representation
        trace_output = output_capture.getvalue().lower()
        graph_representation = str(traced_model.graph)
        
        # Integrated issue detection for both trace warnings and graph operations
        onnx_issue_patterns = {
            # Warning patterns from trace output
            "dynamic": {
                "source": "trace_warning",
                "message": "Dynamic operations or control flow detected",
                "suggestion": "These often cause problems with ONNX export. Consider replacing with static operations.",
                "related_ops": ["prim::If", "aten::_shape_as_tensor"]
            },
            "shape": {
                "source": "trace_warning",
                "message": "Shape inference issues",
                "suggestion": "Shape-dependent operations may not export correctly. Use fixed shapes when possible.",
                "related_ops": ["aten::_shape_as_tensor", "aten::reshape", "aten::view"]
            },
            "not supported": {
                "source": "trace_warning",
                "message": "Potentially unsupported operations",
                "suggestion": "Some PyTorch operations don't have ONNX equivalents. Consider alternatives if export fails.",
                "related_ops": []
            },
            "fallback": {
                "source": "trace_warning",
                "message": "Operator fallback warnings",
                "suggestion": "Fallback operators may not convert properly to ONNX.",
                "related_ops": []
            },
            "deprecated": {
                "source": "trace_warning",
                "message": "Use of deprecated operations",
                "suggestion": "Replace deprecated operations with their modern equivalents.",
                "related_ops": []
            },
            "tensor size": {
                "source": "trace_warning",
                "message": "Dynamic tensor sizing issues",
                "suggestion": "Dynamic sizing can cause problems. Try to use fixed dimensions.",
                "related_ops": ["aten::_shape_as_tensor", "aten::reshape", "aten::view", "aten::flatten"]
            },
            "converting a tensor to a python boolean": {
                "source": "trace_warning",
                "message": "Tensor dimension used in conditional statement",
                "suggestion": "Avoid using tensor dimensions in conditionals. Consider using torch.jit.script instead of trace, or restructure your model to avoid dynamic control flow.",
                "related_ops": ["prim::If"]
            },
            
            # Graph operation patterns
            "aten::unbind": {
                "source": "graph_op",
                "message": "Dynamic slicing operation detected",
                "suggestion": "Dynamic slicing may not translate well to ONNX. Consider alternative approaches."
            },
            "aten::slice": {
                "source": "graph_op",
                "message": "Slice operation detected",
                "suggestion": "Dynamic slicing may not translate well to ONNX. Ensure indices are fixed."
            },
            "prim::listconstruct": {
                "source": "graph_op",
                "message": "Dynamic list creation detected",
                "suggestion": "Dynamic list operations often cause ONNX export issues. Use fixed-size tensors when possible."
            },
            "aten::permute": {
                "source": "graph_op",
                "message": "Tensor permutation detected",
                "suggestion": "Permutations can be tricky in ONNX. Verify the exported model handles these correctly."
            },
            "aten::flatten": {
                "source": "graph_op",
                "message": "Flatten operation detected",
                "suggestion": "Flatten operations may cause shape issues in ONNX. Consider using Reshape with fixed dimensions."
            },
            "aten::_shape_as_tensor": {
                "source": "graph_op",
                "message": "Shape-dependent operation detected",
                "suggestion": "Shape-dependent operations often cause ONNX export issues. Try to use fixed shapes."
            },
            "prim::if": {
                "source": "graph_op",
                "message": "Dynamic control flow detected",
                "suggestion": "Conditional branches based on input dimensions often fail in ONNX export. Refactor to avoid dynamic control flow."
            }
        }
        
            # Ultra-concise single-loop detection
        
        for pattern, info in onnx_issue_patterns.items():
            if (info["source"] == "trace_warning" and trace_output and pattern in trace_output):
                issues.append({
                    "message": f"JIT tracing warning: {info['message']}",
                    "suggestion": info["suggestion"],
                    "detail": trace_output.strip()
                })
            elif (info["source"] == "graph_op" and pattern.lower() in graph_representation.lower()):
                issues.append({
                    "message": info["message"],
                    "suggestion": info["suggestion"]
                })
            
    except Exception as e:
        # If tracing fails, that's often a sign that ONNX conversion will fail too
        issues.append({
            "message": f"Tracing failed: {str(e)}",
            "suggestion": "The model may have dynamic behavior that can't be traced, which could also cause ONNX export issues"
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


def test_onnx_compatibility(model, input_shape=None, batch_size=1):
    """
    Test a PyTorch model for ONNX compatibility and print results.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
    """
    # Run static code analysis first
    static_issues = check_model_code(model, input_shape, batch_size)
    
    # Then run conversion and inference tests
    results = check_onnx_conversion(model, input_shape, batch_size)
    
    # Add static issues separately
    results["static_issues"] = static_issues
    
    # Print results
    print_test_results(results)
    
    return results


# # Example MNIST usage
# if __name__ == "__main__":
#     # Import the MNIST model from separate file
#     from mnist_model import MNISTClassifier
    
#     # Create and test the model
#     model = MNISTClassifier()
    
#     # MNIST images are 1x28x28
#     test_onnx_compatibility(model, input_shape=(1, 28, 28))

# Example problematic model 1 usage
if __name__ == "__main__":
    # Import the MNIST model from separate file
    from problematic_model_1 import DetectableIssuesModel
    
    # Create and test the model
    model = DetectableIssuesModel()
    
    # MNIST images are 1x28x28
    test_onnx_compatibility(model, input_shape=(3, 32, 32))
