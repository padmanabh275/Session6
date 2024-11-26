import torch
import torch.nn as nn
from model import MNIST_DNN
import pytest

def test_parameter_count():
    model = MNIST_DNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_batch_norm_usage():
    model = MNIST_DNN()
    has_batch_norm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"

def test_dropout_usage():
    model = MNIST_DNN()
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_input_output_shape():
    model = MNIST_DNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape should be (1, 10), got {output.shape}"

def test_gap_or_fc_usage():
    model = MNIST_DNN()
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    
    # Check if model uses either GAP or FC layers
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected layers"
    
    # If using GAP, check for proper dimensionality reduction
    if has_gap:
        gap_layer = next(m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d))
        assert gap_layer.output_size == (1, 1), "GAP should reduce spatial dimensions to 1x1"
    
    # If using FC, check for proper flattening
    if has_fc:
        fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(fc_layers) > 0, "Should have at least one FC layer"
        assert fc_layers[-1].out_features == 10, "Final FC layer should output 10 classes"

if __name__ == "__main__":
    pytest.main([__file__]) 