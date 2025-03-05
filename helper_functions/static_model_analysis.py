import torch
import inspect
import warnings
import re

def static_model_analysis(model, input_shape=(3, 224, 224), batch_size=1):
    """
    Focused static analysis of PyTorch model for critical ONNX conversion issues.
    
    Args:
        model: PyTorch model to analyze
        input_shape: Shape of expected input tensor (excluding batch dimension)
        batch_size: Batch size for test input
    
    Returns:
        Dictionary with model_name and list of high-confidence issues
    """
    result = {
        "model_name": type(model).__name__,
        "issues": []
    }
    
    seen_warnings = set()  # Track unique warnings to avoid duplicates
    
    # Analyze model structure first to prioritize critical issues
    model_modules = {name: module for name, module in model.named_modules()}
    
    # Step 1: Look for conditional statements based directly on input dimensions
    for name, module in model_modules.items():
        if not hasattr(module, 'forward') or not callable(module.forward):
            continue
            
        try:
            source = inspect.getsource(module.forward)
            
            # Look specifically for conditional statements acting on input dimensions
            input_condition_pattern = re.compile(
                r'if\s+.*?(?:x\.shape|input\.shape|x\.size\(\)|x\.dim\(\))'
            )
            
            if input_condition_pattern.search(source):
                result["issues"].append({
                    "message": f"Dynamic control flow based on input dimensions in {name}.forward()",
                    "suggestion": "Replace conditional statements depending on input dimensions with static logic"
                })
            
            # Look for dynamic reshaping using variable dimensions
            dynamic_reshape_pattern = re.compile(
                r'(?:\.view|\.reshape|\.flatten)\(.*?(?:shape\[|size\(|\.dim)'
            )
            
            if dynamic_reshape_pattern.search(source):
                result["issues"].append({
                    "message": f"Dynamic reshaping with variable dimensions in {name}.forward()",
                    "suggestion": "Use fixed dimensions in reshape/view operations"
                })
            
            # Only flag specific problematic operations that are known to cause issues
            high_risk_ops = {
                r'torch\.unbind\(': "Unbind operation",
                r'torch\.where\(.*?\.shape': "Dynamic conditional operation using shapes",
                r'torch\.nonzero\(': "Nonzero operation", 
                r'\.scatter\(': "Scatter operation",
                r'torch\.topk\(.*?\d+.*?\)': "TopK with dynamic k value"
            }
            
            for pattern, message in high_risk_ops.items():
                op_pattern = re.compile(pattern)
                if op_pattern.search(source):
                    result["issues"].append({
                        "message": f"{message} in {name}.forward() that frequently fails in ONNX conversion",
                        "suggestion": "Replace with ONNX-compatible alternatives"
                    })
            
        except Exception:
            # Skip modules we can't inspect without raising errors
            pass
    
    # Step 2: Test tracing to catch confirmation of critical issues
    try:
        dummy_input = torch.randn(batch_size, *input_shape)
        model.eval()  # Ensure model is in evaluation mode
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Use script mode which is more thorough in catching dynamic issues
            try:
                torch.jit.script(model)
            except:
                # If script fails, try trace
                torch.jit.trace(model, dummy_input)
            
            # Filter for critical warnings that strongly indicate conversion will fail
            critical_patterns = [
                "Converting a tensor to a Python boolean",
                "Cannot determine the shape statically",
                "has a dynamic dimension",
                "Expected isFloatingPoint for argument",
                "Expected a constant value"
            ]
            
            for warning in w:
                warning_msg = str(warning.message)
                # Deduplicate similar warnings
                warning_key = warning_msg[:100]  # Use first 100 chars as fingerprint
                
                if warning_key in seen_warnings:
                    continue
                    
                seen_warnings.add(warning_key)
                
                # Only report warnings that are highly indicative of conversion failures
                if any(pattern in warning_msg for pattern in critical_patterns):
                    result["issues"].append({
                        "message": f"Critical tracing issue: {warning_msg}",
                        "suggestion": "This warning indicates potential ONNX conversion failure"
                    })
    
    except Exception as e:
        # If tracing completely fails, that's a strong indicator
        error_msg = str(e)
        
        # Check for common error patterns that definitely mean conversion will fail
        if any(pattern in error_msg for pattern in [
            "Expected a Tensor",
            "Expected isFloatingPoint",
            "Expected a constant",
            "dictionary construction",
            "Cannot determine the shape",
            "Cannot call a value"
        ]):
            result["issues"].append({
                "message": f"Model tracing failed: {error_msg}",
                "suggestion": "This error confirms the model will likely fail ONNX conversion"
            })
    
    # Limit number of issues to avoid overwhelming output
    if len(result["issues"]) > 5:
        result["issues"] = result["issues"][:5]
    
    return result