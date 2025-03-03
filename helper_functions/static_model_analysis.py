import torch
import io
from contextlib import redirect_stdout, redirect_stderr


def static_model_analysis(model, input_shape=(3, 32, 32), batch_size=1):
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