# InstGS: Instantaneous Gaussian Splatting via Cross-Frame Instance Segmentation

3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have demonstrated remarkable capabilities in photo-realistic novel view synthesis. However, their practical adoption is often hindered by substantial storage requirements and limited rendering efficiency, particularly for scenes with repetitive structures. While existing acceleration methods primarily focus on optimizing individual Gaussian primitives or neural network architectures, they fail to address the fundamental redundancy inherent in repetitive content.

To overcome this limitation, we introduce **InstGS**, the first Gaussian instancing-based accelerated rendering framework. To eliminate redundancy at the representation level, we perform gradient-driven cross-frame instance segmentation to group similar Gaussians into reusable components. A shared Gaussian template with instance-specific offsets is optimized to replace all similar instances, yielding substantial memory saving with negligible loss in visual fidelity. Extensive experiments demonstrate that InstGS achieves high-quality, high-frame-rate, and low-memory rendering performance.

> Built upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Inria GRAPHDECO.

## Key Features

- **Gaussian Instancing** — Reuse shared Gaussian templates across repetitive scene structures, dramatically reducing storage while maintaining visual quality
- **Cross-Frame Instance Segmentation** — Gradient-driven grouping of similar Gaussians across views into reusable components
- **Instance-Specific Offsets** — Per-instance deformable offsets (position, color, opacity) applied to shared templates for scene adaptation
- **Custom CUDA Rasterizer** — High-performance instance Gaussian rasterizer with native instancing support (`inst-gaussian-rasterization`)
- **Standard 3DGS Baseline** — Full training pipeline with depth regularization and anti-aliasing (`train.py`)

## Installation

### Prerequisites

- Ubuntu 22.04 (or compatible Linux)
- CUDA 12.8
- Python 3.10
- PyTorch 2.5+

### Setup

```bash
# Clone with submodules
git clone --recursive git@github.com:Strange-tech/InstGS.git
cd InstGS

# Install dependencies
pip install -r requirements.txt  # if available, or install manually:
pip install torch torchvision plyfile tqdm open3d scipy lpips torchmetrics pytorch3d einops

# Build submodules
# diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# inst-gaussian-rasterization (custom CUDA rasterizer)
pip install submodules/inst-gaussian-rasterization

# simple-knn
pip install submodules/simple-knn

# fused-ssim (optional, for faster SSIM)
pip install submodules/fused-ssim
```

## Usage

### 1. Preprocess images with COLMAP

```bash
python convert.py -s /path/to/your/scene
```

Or use the shell script:

```bash
bash run_colmap.sh /path/to/your/scene
```

### 2. Training

**Standard 3DGS training:**
```bash
python train.py -s /path/to/scene -m /path/to/output
```

**Instance-aware training:**
```bash
python train_instances.py -s /path/to/scene -m /path/to/output
```

**Instance-aware training (CUDA-accelerated):**
```bash
python train_instances_cuda.py -s /path/to/scene -m /path/to/output
```

### 3. Rendering

```bash
# Render instances
python render_instances.py -s /path/to/scene -m /path/to/model

# Render instances (CUDA-accelerated)
python render_instances_cuda.py -s /path/to/scene -m /path/to/model
```

## Project Structure

```
.
├── arguments/              # Argument parsing (Model, Pipeline, Optimization params)
├── gaussian_renderer/      # Core rendering logic (standard + instanced)
├── lpipsPyTorch/           # LPIPS perceptual metric
├── scene/                  # Scene loading, Gaussian models, camera utils
│   ├── gaussian_model.py       # Standard Gaussian model
│   └── inst_gaussian_model.py  # Instance-aware Gaussian model
├── submodules/             # Git submodules
│   ├── diff-gaussian-rasterization/  # Original CUDA rasterizer
│   ├── inst-gaussian-rasterization/  # Custom instance CUDA rasterizer
│   ├── simple-knn/                  # Simple KNN for point cloud init
│   └── fused-ssim/                  # Fused SSIM implementation
├── utils/                  # Utility functions (graphics, images, loss, etc.)
├── train.py                # Standard 3DGS training
├── train_instances.py      # Instance-aware training
├── train_instances_cuda.py # Instance-aware training (CUDA-accelerated)
├── render_instances.py     # Instance rendering
├── render_instances_cuda.py# Instance rendering (CUDA-accelerated)
├── convert.py              # COLMAP preprocessing
└── metrics.py              # Evaluation metrics
```

## Assets & Figures

If you have result figures, teaser images, or architecture diagrams, place them in an `assets/` directory at the project root:

```
assets/
├── teaser.png              # Main teaser figure
├── architecture.png        # Method architecture diagram
├── results/                # Qualitative comparison results
│   ├── scene1_baseline.png
│   ├── scene1_ours.png
│   └── ...
└── videos/                 # Demo videos (optional)
```

Reference these images in your README like so:
```markdown
![Teaser](assets/teaser.png)
```

## Submodules

This project uses the following git submodules. Make sure to clone with `--recursive` or run:

```bash
git submodule update --init --recursive
```

| Submodule | Description | Upstream |
|-----------|-------------|----------|
| `diff-gaussian-rasterization` | CUDA rasterizer for 3DGS | [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) |
| `inst-gaussian-rasterization` | Custom CUDA rasterizer for instance GS | Forked & modified |
| `simple-knn` | Simple KNN for point cloud initialization | [bkerbl/simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) |
| `fused-ssim` | Fused SSIM for faster training | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) |

## License

This project includes code from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Inria GRAPHDECO, which is free for **non-commercial, research and evaluation use** under the terms of the LICENSE.md file.

For inquiries about the original 3DGS, contact: george.drettakis@inria.fr

## Citation

If you use InstGS in your research, please cite:

```bibtex
@article{lu2025instgs,
  author    = {Lu, Ziang and { }},
  title     = {InstGS: Instantaneous Gaussian Splatting via Cross-Frame Instance Segmentation},
  year      = {2025},
}
```

The original 3D Gaussian Splatting:

```bibtex
@article{kerbl20233dgs,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  volume    = {42},
  number    = {4},
  year      = {2023},
}
```
