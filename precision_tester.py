import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass
import logging

@dataclass
class PrecisionTestConfig:
    """Configuration for precision testing"""
    rtol: float = 1e-3  # Relative tolerance
    atol: float = 1e-5  # Absolute tolerance
    input_shapes: List[Tuple[int, ...]] = None  # List of input shapes to test
    test_cases: int = 100  # Number of test cases per shape
    data_ranges: Dict[str, Tuple[float, float]] = None  # Input data ranges

class PrecisionTester:
    def __init__(self, original_model: nn.Module, onnx_path: str, config: PrecisionTestConfig):
        self.original_model = original_model
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
    def generate_test_data(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate random test data with specified shape and range"""
        if self.config.data_ranges and 'input' in self.config.data_ranges:
            min_val, max_val = self.config.data_ranges['input']
        else:
            min_val, max_val = -1.0, 1.0
            
        return torch.rand(shape) * (max_val - min_val) + min_val
        
    def compare_outputs(self, torch_output: torch.Tensor, onnx_output: np.ndarray) -> Dict[str, float]:
        """Compare outputs and calculate various metrics"""
        torch_output_np = torch_output.detach().numpy()
        
        # Calculate absolute differences
        abs_diff = np.abs(torch_output_np - onnx_output)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # Calculate relative differences
        rel_diff = abs_diff / (np.abs(torch_output_np) + 1e-10)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Check if outputs are within tolerance
        is_close = np.allclose(
            torch_output_np, 
            onnx_output, 
            rtol=self.config.rtol, 
            atol=self.config.atol
        )
        
        return {
            'max_absolute_diff': float(max_abs_diff),
            'mean_absolute_diff': float(mean_abs_diff),
            'max_relative_diff': float(max_rel_diff),
            'mean_relative_diff': float(mean_rel_diff),
            'within_tolerance': bool(is_close)
        }
        
    def run_single_test(self, input_shape: Tuple[int, ...]) -> Dict[str, Union[float, bool]]:
        """Run a single test with given input shape"""
        # Generate test data
        test_input = self.generate_test_data(input_shape)
        
        # Get PyTorch output
        self.original_model.eval()
        with torch.no_grad():
            torch_output = self.original_model(test_input)
            
        # Get ONNX output
        onnx_input = {
            self.onnx_session.get_inputs()[0].name: test_input.numpy()
        }
        onnx_output = self.onnx_session.run(None, onnx_input)[0]
        
        # Compare outputs
        return self.compare_outputs(torch_output, onnx_output)
        
    def run_tests(self) -> Dict[str, List[Dict[str, Union[float, bool]]]]:
        """Run all tests and collect results"""
        results = {}
        
        for shape in self.config.input_shapes:
            shape_results = []
            self.logger.info(f"Testing input shape: {shape}")
            
            for i in range(self.config.test_cases):
                if i % 10 == 0:
                    self.logger.debug(f"Running test case {i+1}/{self.config.test_cases}")
                    
                test_result = self.run_single_test(shape)
                shape_results.append(test_result)
                
                if not test_result['within_tolerance']:
                    self.logger.warning(
                        f"Test case {i+1} for shape {shape} exceeded tolerance limits:\n"
                        f"Max absolute difference: {test_result['max_absolute_diff']}\n"
                        f"Max relative difference: {test_result['max_relative_diff']}"
                    )
            
            results[str(shape)] = shape_results
            
        return results

def analyze_results(results: Dict[str, List[Dict[str, Union[float, bool]]]]) -> Dict[str, Dict[str, float]]:
    """Analyze test results and provide summary statistics"""
    summary = {}
    
    for shape, shape_results in results.items():
        # Calculate statistics across all test cases for this shape
        max_abs_diffs = [r['max_absolute_diff'] for r in shape_results]
        max_rel_diffs = [r['max_relative_diff'] for r in shape_results]
        within_tolerance_count = sum(r['within_tolerance'] for r in shape_results)
        
        summary[shape] = {
            'worst_absolute_diff': max(max_abs_diffs),
            'mean_absolute_diff': np.mean(max_abs_diffs),
            'worst_relative_diff': max(max_rel_diffs),
            'mean_relative_diff': np.mean(max_rel_diffs),
            'passing_rate': within_tolerance_count / len(shape_results)
        }
    
    return summary

# Example usage:
if __name__ == "__main__":
    import torch.onnx

    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 10)
    
    # Export the model to ONNX
    onnx_path = "model.onnx"
    torch.onnx.export(
        model,               # PyTorch model
        dummy_input,        # Dummy input
        onnx_path,          # Output path
        export_params=True,  # Store the trained parameters
        opset_version=11,   # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding
        input_names=['input'],     # Names for the input and output nodes
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")
    
    # Configure test parameters
    config = PrecisionTestConfig(
        rtol=1e-3,
        atol=1e-5,
        input_shapes=[(32, 10), (64, 10), (128, 10)],
        test_cases=100,
        data_ranges={'input': (-2.0, 2.0)}
    )
    
    # Initialize tester with the exported model
    tester = PrecisionTester(model, onnx_path, config)
    
    # Run tests
    results = tester.run_tests()
    
    # Analyze results
    summary = analyze_results(results)
    
    # Print summary
    for shape, stats in summary.items():
        print(f"\nResults for input shape {shape}:")
        print(f"Passing rate: {stats['passing_rate']*100:.2f}%")
        print(f"Worst absolute difference: {stats['worst_absolute_diff']:.2e}")
        print(f"Mean absolute difference: {stats['mean_relative_diff']:.2e}")