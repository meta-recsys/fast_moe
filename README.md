# FastMoE - Fast Mixture of Experts for PyTorch

FastMoE is a high-performance PyTorch library for Mixture of Experts (MoE) layers, featuring optimized GPU kernels implemented in both PyTorch and Triton. It provides efficient expert parallelism and supports FP8 quantization for faster inference and training on modern NVIDIA GPUs.

## Features

- **Multiple Kernel Backends**: Choose between PyTorch native kernels or optimized Triton kernels
- **FP8 Quantization Support**: Leverage FP8 precision for faster computation on H100 GPUs
- **Expert Parallelism**: Distributed expert computation across multiple GPUs
- **Optimized Operations**: Fused kernels for index_select + BMM, SiLU activation, and more
- **Flexible Routing**: Support for Top-K expert selection with customizable gating networks
- **Production Ready**: Extensively tested with property-based testing using Hypothesis
- **Performance Benchmarking**: Built-in benchmarking tools for comparing kernel implementations

## Architecture Overview

FastMoE implements a highly optimized Mixture of Experts layer with the following key components:

- **Gating Network**: Routes tokens to top-k experts based on learned scores
- **Expert Layers**: Parallel feed-forward networks specialized for different input patterns
- **Optimized Kernels**: Custom CUDA/Triton kernels for batched operations on variable-length sequences
- **Load Balancing**: Efficient token routing with support for load balancing losses

## Requirements

### Core Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 12.0 (for GPU acceleration)
- Triton >= 2.1 (for Triton kernels)

### Optional Dependencies

- fbgemm_gpu (Meta's GPU GEMM library - currently internal)
- mslk (Meta's machine learning kernel library - currently internal)

**Note**: This repository currently has dependencies on Meta-internal libraries (`fbgemm_gpu`, `mslk`). These dependencies are being evaluated for replacement with open-source alternatives or will be made optional in future releases.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/facebookresearch/fast_moe.git
cd fast_moe

# Install dependencies
pip install -r requirements.txt

# Install FastMoE
pip install -e .
```

**Note**: The current build system uses Buck2 (Meta's internal build system). A standard Python packaging setup (setup.py/pyproject.toml) is being added for open-source distribution.

## Quick Start

### Basic Usage

```python
import torch
from fast_moe.modules.fast_moe_module import FastMoELayer
from fast_moe.kernels.utils import KernelType

# Create a FastMoE layer
moe_layer = FastMoELayer(
    input_dim=1024,      # Input feature dimension
    hidden_dim=4096,     # Hidden layer dimension
    output_dim=1024,     # Output feature dimension
    num_experts=8,       # Number of expert networks
    kernel=KernelType.TRITON  # Use Triton kernels (or KernelType.PYTORCH)
)

# Move to GPU
moe_layer = moe_layer.cuda()

# Forward pass
batch_size, seq_len = 4, 512
x = torch.randn(batch_size, seq_len, 1024, device='cuda')
num_experts_per_tok = 2  # Each token routes to top-2 experts

output = moe_layer(x, num_experts_per_tok=num_experts_per_tok)
# output shape: [batch_size, seq_len, output_dim]
```

### Using Different Kernel Types

```python
from fast_moe.kernels.utils import KernelType

# PyTorch native kernels (compatible with all GPUs)
moe_pytorch = FastMoELayer(..., kernel=KernelType.PYTORCH)

# Triton optimized kernels (recommended for A100/H100)
moe_triton = FastMoELayer(..., kernel=KernelType.TRITON)
```

### Expert Parallel Training

```python
from fast_moe.modules.expert_parallel_fast_moe import ExpertParallelFastMoELayer

# Distributed MoE with expert parallelism
ep_moe_layer = ExpertParallelFastMoELayer(
    input_dim=1024,
    hidden_dim=4096,
    output_dim=1024,
    num_experts=64,  # Experts distributed across GPUs
    expert_parallel_size=8  # Number of GPUs for expert parallelism
)
```

## Kernel Types

FastMoE supports multiple kernel implementations optimized for different scenarios:

| Kernel Type | Backend | Best For | Requirements |
|-------------|---------|----------|--------------|
| `PYTORCH` | PyTorch Native | Compatibility, CPU fallback | PyTorch >= 2.0 |
| `TRITON` | Triton GPU Kernels | A100/H100 GPUs, maximum performance | Triton >= 2.1, CUDA GPU |
| `FP8` | Triton FP8 | H100 GPUs with FP8 support | H100 GPU, CUDA >= 12.0 |

## Performance

FastMoE is designed for high-performance MoE computation with the following optimizations:

- **Fused Operations**: Combined index selection, matrix multiplication, and activation functions
- **Jagged Tensor Operations**: Efficient handling of variable-length expert assignments
- **Memory Efficiency**: Optimized memory layout for GPU computation
- **Kernel Fusion**: Reduced memory traffic through operator fusion

### Benchmarking

Run the built-in benchmarks to measure performance on your hardware:

```bash
# Full benchmark suite
python benchmarks/fast_moe_benchmark.py

# Quick benchmark
python benchmarks/fast_moe_benchmark.py --quick

# Profile specific kernel
python benchmarks/fast_moe_benchmark.py --profile triton
```

## Project Structure

```
fast_moe/
├── kernels/              # Core kernel implementations
│   ├── moe.py           # Main kernel dispatcher
│   ├── moe_fp8.py       # FP8 quantization kernels
│   ├── pytorch/         # PyTorch native implementations
│   ├── triton/          # Triton GPU kernels
│   ├── tests/           # Kernel unit tests
│   └── benchmarks/      # Kernel-level benchmarks
├── modules/             # High-level MoE modules
│   ├── fast_moe_module.py              # Main FastMoELayer
│   ├── expert_parallel_fast_moe.py     # Distributed expert parallelism
│   └── tests/           # Module-level tests
├── utils/               # Utilities and configurations
│   ├── enums.py        # Enumerations (ExpertType, RouterChoice, etc.)
│   ├── configs.py      # Configuration dataclasses
│   └── utils.py        # Helper functions
└── benchmarks/          # High-level benchmarks
```

## Testing

FastMoE includes comprehensive unit tests using property-based testing with Hypothesis:

```bash
# Run all tests
python -m pytest kernels/tests/
python -m pytest modules/tests/

# Run specific test
python -m pytest kernels/tests/moe_test.py -v

# Test with different GPU architectures
python -m pytest kernels/tests/moe_test.py --gpu h100
```

**Note**: Most tests require NVIDIA GPU with CUDA support (A100 or H100 recommended).

## API Reference

### FastMoELayer

The main MoE layer implementation.

```python
FastMoELayer(
    input_dim: int,           # Input feature dimension
    hidden_dim: int,          # Hidden layer dimension
    output_dim: int,          # Output feature dimension
    num_experts: int,         # Number of expert networks
    kernel: KernelType        # Kernel backend to use
)
```

**Methods**:
- `forward(x: Tensor, num_experts_per_tok: int) -> Tensor`: Forward pass with token routing

### KernelType (Enum)

```python
from fast_moe.kernels.utils import KernelType

KernelType.PYTORCH  # PyTorch native kernels
KernelType.TRITON   # Triton optimized kernels
```

### ExpertType (Enum)

```python
from fast_moe.utils.enums import ExpertType

ExpertType.MLP      # Multi-layer perceptron experts
ExpertType.FFN      # Feed-forward network experts
ExpertType.FFNBIAS  # FFN with bias
```

### RouterChoice (Enum)

```python
from fast_moe.utils.enums import RouterChoice

RouterChoice.TopK     # Top-K expert selection
RouterChoice.Vanilla  # Standard routing
```

## Development Status & Roadmap

### Current Status

FastMoE is currently in active development. This open-source release is based on production code used internally at Meta.

### Known Limitations

- **Internal Dependencies**: Some features require Meta-internal libraries (`fbgemm_gpu`, `mslk`)
- **Build System**: Currently uses Buck2; standard Python packaging in progress
- **GPU Requirements**: Most kernels optimized for NVIDIA A100/H100 GPUs

### Roadmap

- [ ] Replace internal dependencies with open-source alternatives
- [ ] Add standard Python packaging (setup.py/pyproject.toml)
- [ ] CPU fallback implementations
- [ ] Integration examples with Hugging Face Transformers
- [ ] Additional routing strategies (e.g., expert choice routing)
- [ ] Support for more GPU architectures
- [ ] Pre-trained model checkpoints

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/facebookresearch/fast_moe.git
cd fast_moe

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Style

This project uses:
- `black` for code formatting
- `isort` for import sorting
- `mypy` for type checking
- `flake8` for linting

## Citation

If you use FastMoE in your research, please cite:

```bibtex
@software{fastmoe2024,
  title = {FastMoE: High-Performance Mixture of Experts for PyTorch},
  author = {Meta AI Research},
  year = {2024},
  url = {https://github.com/facebookresearch/fast_moe}
}
```

## License

FastMoE is released under the [MIT License](LICENSE). See the LICENSE file for details.

## Related Projects

- [MegaBlocks](https://github.com/databricks/megablocks) - Efficient sparse MoE implementation
- [Tutel](https://github.com/microsoft/tutel) - Highly optimized MoE library
- [DeepSpeed-MoE](https://github.com/microsoft/DeepSpeed) - MoE support in DeepSpeed
- [Fairseq MoE](https://github.com/facebookresearch/fairseq) - MoE in Fairseq

## Support

- **Issues**: Please report bugs and feature requests on [GitHub Issues](https://github.com/facebookresearch/fast_moe/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/facebookresearch/fast_moe/discussions)
- **Documentation**: Full documentation available at [docs.fast-moe.ai](https://docs.fast-moe.ai) (coming soon)

## Acknowledgments

FastMoE is developed and maintained by Meta AI Research. We thank the PyTorch team for their excellent framework and the Triton team for their kernel compiler.

Special thanks to the contributors of MegaBlocks and other open-source MoE implementations for inspiration and benchmarking comparisons.
