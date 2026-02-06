# Quick Start Guide

Get up and running with RRF reconstruction in under 10 minutes!

## Prerequisites

- Ubuntu 20.04+ (or WSL2 on Windows)
- NVIDIA GPU (8GB+ VRAM)
- CUDA 11.8+
- Conda or Python 3.8+

## Installation (5 minutes)

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline.git
cd RRF-Reconstruction-Pipeline

# 2. Create conda environment
conda create -n rrf-pipeline python=3.8 -y
conda activate rrf-pipeline

# 3. Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install TensorFlow GPU
pip install tensorflow[and-cuda]==2.15.0

# 5. Install dependencies
pip install -r requirements.txt

# 6. Clone RF-3DGS framework
git clone https://github.com/Wangmz-1203/RF-3DGS.git
cd RF-3DGS

# 7. Install RF-3DGS submodules
cd submodules
pip install ./diff-surfel-rasterization
pip install ./simple-knn
cd ../..

# 8. Verify installation
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('TensorFlow GPU:', len(tf.config.list_physical_devices('GPU')))"
python -c "from sionna.rt import load_scene; print('Sionna: OK')"
```

## Quick Demo (5 minutes)

### Step 1: Generate 3D Scene (10 seconds)

```bash
python create_scene_5x3x3_multi.py
```

**Output**: `meshes/` folder with 7 PLY files + `room_5x3x3_combined.ply`

### Step 2: Generate Visual Dataset (2-3 hours)

```bash
# Install Blender first (if not already installed)
sudo snap install blender --classic

# Generate 300 RGB images
blender --background --python generate_visual_dataset.py
```

**Output**: `dataset_visual_v2/` with 300 images + camera poses

**⏩ Skip this step**: Download pre-generated dataset from [releases](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/releases)

### Step 3: Generate RF Dataset (2-3 hours)

```bash
python generate_dataset_ideal_mpc.py
```

**Output**: `dataset_custom_scene_ideal_mpc/` with RF heatmaps

**⏩ Skip this step**: Download pre-generated dataset from [releases](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/releases)

### Step 4: Train RRF Model (3-6 hours)

```bash
cd RF-3DGS

# Run full training pipeline (Stage 1 + Stage 2)
bash run_rf_reconstruction.sh
```

**Output**: 
- `output/visual_model/` - Visual reconstruction
- `output/rf_model/` - RF reconstruction

**What's happening**:
- Stage 1 (2-4 hours): Learn geometry from RGB images
- Stage 2 (1-2 hours): Learn RF patterns from heatmaps

### Step 5: Evaluate Results (1 minute)

```bash
# Render test views
python render.py -m output/rf_model --iteration 10000

# Compute metrics
python metrics.py -m output/rf_model
```

**Expected Output**:
```
PSNR:  27.34 ± 1.87 dB
SSIM:  0.891 ± 0.032
LPIPS: 0.187 ± 0.042
```

### Step 6: Visualize (Interactive)

```bash
cd SIBR_viewers

# Build viewer (first time only)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Launch viewer
./build/bin/SIBR_gaussianViewer_app -m ../output/rf_model --iteration 10000
```

**Controls**: Mouse to rotate, WASD to move, Q/E for up/down

---

## Minimal Example (with pre-trained model)

If you just want to see the results without training:

```bash
# 1. Download pre-trained model
wget https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/releases/download/v1.0.0/rf_model_pretrained.zip
unzip rf_model_pretrained.zip -d RF-3DGS/output/

# 2. Render test views
cd RF-3DGS
python render.py -m output/rf_model --iteration 10000

# 3. Launch viewer
cd SIBR_viewers
./build/bin/SIBR_gaussianViewer_app -m ../output/rf_model --iteration 10000
```

---

## Troubleshooting

### CUDA Not Found
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install:
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Out of Memory
```bash
# Reduce resolution in scripts:
# In generate_visual_dataset.py:
RESOLUTION = 640  # Instead of 800

# In train.py:
python train.py --resolution 2  # Half resolution
```

### Blender Not Found
```bash
# Install Blender
sudo snap install blender --classic

# Or download from: https://www.blender.org/download/
```

### Sionna Import Error
```bash
# Reinstall Sionna
pip uninstall sionna
pip install sionna --no-cache-dir

# Verify
python -c "from sionna.rt import load_scene; print('OK')"
```

---

## Time Estimates

| Step | Time | Can Skip? |
|------|------|-----------|
| Installation | 5 min | No |
| Scene Creation | 10 sec | No |
| Visual Dataset | 2-3 hours | Yes (download) |
| RF Dataset | 2-3 hours | Yes (download) |
| Training | 3-6 hours | Yes (use pre-trained) |
| Evaluation | 1 min | No |
| Visualization | Instant | No |

**Total**: 6-12 hours (or 10 minutes with pre-trained model)

---

## Next Steps

After completing the quick start:

1. **Customize Scene**: Edit `create_scene_5x3x3_multi.py` to create your own room
2. **Experiment**: Try different RF frequencies, transmitter positions
3. **Read Docs**: See `docs/` for detailed explanations
4. **Contribute**: Open issues, submit PRs on GitHub

---

## Common Questions

**Q: Can I use CPU only?**  
A: No, GPU is required. CPU-only will be 100x slower and impractical.

**Q: What GPU do I need?**  
A: Minimum: GTX 1080 (8GB). Recommended: RTX 3080 or better.

**Q: Can I use Windows?**  
A: Yes, via WSL2 (Windows Subsystem for Linux). Native Windows is not tested.

**Q: How much disk space?**  
A: ~50GB total (10GB datasets + 20GB models + 20GB outputs)

**Q: Can I use this for real-world scenes?**  
A: Yes! Replace synthetic dataset with COLMAP reconstruction from real images.

---

## Support

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/issues)
- 💬 [Discussions](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/discussions)
- 📧 Email: your_email@example.com

---

**Happy Reconstructing!** 🎉
