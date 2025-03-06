import torch
import onnx

def convert_to_onnx(model, input_shape=(3, 224, 224), batch_size=1):
    """
    Convert a PyTorch model to ONNX format and check for conversion issues.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for test input
        
    Returns:
        Dictionary with conversion results and issue information
    """
    # Set model to evaluation mode
    model.eval()
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create random input tensor
    input_tensor = torch.randn((batch_size,) + input_shape).to(device)
    
    # Initialize results
    results = {
        "model_name": type(model).__name__,
        "conversion_success": False,
        "issues": [],
        "onnx_path": "temp_model.onnx",
        "device": device
    }
    
    # Step 1: Convert model to ONNX
    try:
        torch.onnx.export(
            model,
            input_tensor,
            results["onnx_path"],
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
        onnx.checker.check_model(onnx.load(results["onnx_path"]))
        results["conversion_success"] = True
        
    except Exception as e:
        error_msg = str(e)
        results["conversion_success"] = False
        results["issues"].append({
            "type": "conversion_error",
            "message": f"Conversion failed: {error_msg}"
        })

        # Organize error patterns according to Fix-Con fault categories
        error_patterns = {
            # Input-based errors: Preprocessing (PP)
            "normalization": "Preprocessing mismatch (check normalization values)",
            "scale": "Input scaling issue (preprocessing mismatch)",
            
            # Input-based errors: Input Dimensions (ID) 
            "Sizes of tensors must match": "Input dimension mismatch between source and target",
            "shape": "Input shape inconsistency (dimension ordering issue)",
            "dimension": "Dimension mismatch in operations",
            
            # Input-based errors: Tensor Shape and Structure (TSS)
            "transpose": "Tensor structure issue (incorrect dimension ordering)",
            "permute": "Tensor permutation issue (dimension reordering problem)",
            "dynamic": "Dynamic tensor shape issue",
            "view": "Reshape/view operation issue (structure information loss)",
            "reshape": "Reshape operation issue (structure information loss)",
            
            # Layer-based errors: Weights & Biases (WB)
            "parameter": "Model parameter conversion issue",
            "weight": "Weight tensor conversion issue",
            "bias": "Bias tensor conversion issue",
            "precision": "Numerical precision issue in weights or biases",
            
            # Layer-based errors: Layer Hyperparameters (LH)
            "padding": "Padding hyperparameter inconsistency",
            "stride": "Stride hyperparameter issue",
            "dilation": "Dilation hyperparameter issue",
            "kernel": "Kernel size hyperparameter issue",
            "epsilon": "Epsilon value mismatch (often in normalization layers)",
            
            # Layer-based errors: Computation Graph (CG)
            "No conversion for operator": "Unsupported operation in computation graph",
            "graph": "Computation graph structure issue",
            "node": "Node conversion issue in computation graph",
            "not supported": "Operation not supported in computation graph",
            "sequential": "Sequential operation issue in computation graph",
            "Converting a tensor to a Python boolean": "Conditional operation in graph (control flow issue)",
            
            # Common operation-specific issues
            "unbind": "Tensor unbind operation issue",
            "scatter": "Scatter/gather operation issue",
            "slice": "Slice operation issue",
            "LSTM": "LSTM/RNN conversion issue",
            "RNN": "RNN conversion issue",
            "GRU": "GRU conversion issue",
            "inplace": "Inplace operation issue",
            "topk": "TopK operation issue",
            "softmax": "Softmax operation issue",
            "pooling": "Pooling operation issue"
        }
        
        # Find matching error pattern
        matched = False
        for pattern, explanation in error_patterns.items():
            if pattern.lower() in error_msg.lower():
                # Determine fault category based on pattern
                category = None
                if pattern in ["normalization", "scale"]:
                    category = "Preprocessing (PP)"
                elif pattern in ["Sizes of tensors must match", "shape", "dimension"]:
                    category = "Input Dimensions (ID)"
                elif pattern in ["transpose", "permute", "dynamic", "view", "reshape"]:
                    category = "Tensor Shape and Structure (TSS)"
                elif pattern in ["parameter", "weight", "bias", "precision"]:
                    category = "Weights & Biases (WB)"
                elif pattern in ["padding", "stride", "dilation", "kernel", "epsilon"]:
                    category = "Layer Hyperparameters (LH)"
                else:
                    category = "Computation Graph (CG)"
                
                results["issues"].append({
                    "fault_category": category,
                    "message": explanation,
                    "suggestion": "Check the model for this specific issue"
                })
                matched = True
        
        # If no specific pattern matched, add a generic error
        if not matched:
            results["issues"].append({
                "fault_category": "Unknown",
                "message": "Conversion error without specific pattern match",
                "suggestion": "Manual inspection required"
            })
              
    return results