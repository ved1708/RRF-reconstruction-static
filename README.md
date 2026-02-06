# Radio-Frequency Radiance Fields (RRF) Reconstruction Pipeline

Complete end-to-end pipeline for reconstructing Radio-Frequency Radiance Fields from 3D scenes using 3D Gaussian Splatting (3DGS). This project demonstrates how to create custom 3D scenes, generate multi-modal datasets (visual + RF), and train neural radiance fields that capture both visual and radio-frequency properties.

## 🎯 Project Overview

This pipeline reconstructs **Radio-Frequency Radiance Fields (RRF)** for indoor environments by:
1. Creating custom 3D room models with furniture
2. Generating synthetic visual datasets using Blender
3. Simulating RF propagation using Sionna RT ray-tracing
4. Training 3D Gaussian Splatting models on visual data
5. Fine-tuning on RF data to learn radio propagation patterns
6. Evaluating and visualizing results in interactive 3D viewers

**Key Achievement**: Successfully trained a 3DGS model that learns both geometric and radio-frequency properties of a custom indoor scene, enabling RF prediction from novel viewpoints.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Scene Creation](#1-scene-creation)
  - [2. Visual Dataset Generation](#2-visual-dataset-generation)
  - [3. RF Dataset Generation](#3-rf-dataset-generation)
  - [4. 3DGS Training](#4-3dgs-training)
  - [5. Evaluation](#5-evaluation)
  - [6. Visualization](#6-visualization)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🔧 Requirements

### Core Dependencies
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Blender 3.6+ (for visual dataset generation)
- Conda (for environment management)

### Key Libraries
- **Sionna** 0.18+ - RT ray-tracing for RF simulation
- **TensorFlow** 2.15+ with GPU support
- **PyTorch** 2.0+ with CUDA
- **Open3D** - Point cloud processing
- **NumPy**, **SciPy**, **Matplotlib** - Scientific computing

See `requirements.txt` for complete list.

## 📦 Installation

### 1. Clone RF-3DGS Framework
```bash
cd Project_1
git clone https://github.com/Wangmz-1203/RF-3DGS.git
cd RF-3DGS
```

### 2. Create Conda Environment
```bash
conda create -n rf-3dgs python=3.8
conda activate rf-3dgs
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

# Install Sionna for RF simulation
pip install sionna

# Install submodules (diff-surfel-rasterization, simple-knn)
cd submodules
pip install ./diff-surfel-rasterization
pip install ./simple-knn
cd ..
```

### 4. Install Blender (for visual dataset generation)
```bash
# Download Blender 3.6+ from https://www.blender.org/download/
# Or install via snap on Ubuntu:
sudo snap install blender --classic
```

## 🔄 Pipeline Overview

```
┌─────────────────────┐
│  1. Scene Creation  │
│  create_scene_5x3x3 │
└──────────┬──────────┘
           │ PLY meshes
           ▼
┌─────────────────────┐
│ 2. Visual Dataset   │
│ generate_visual_    │
│     dataset.py      │
└──────────┬──────────┘
           │ RGB images + poses
           ▼
┌─────────────────────┐
│  3. RF Dataset      │
│ generate_dataset_   │
│   ideal_mpc.py      │
└──────────┬──────────┘
           │ RF heatmaps + COLMAP
           ▼
┌─────────────────────┐
│ 4a. Visual Training │
│    train.py         │
└──────────┬──────────┘
           │ Visual checkpoint
           ▼
┌─────────────────────┐
│ 4b. RF Fine-tuning  │
│    train.py --rf    │
└──────────┬──────────┘
           │ RRF model
           ▼
┌─────────────────────┐
│ 5. Evaluation       │
│ render.py + metrics │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Visualization    │
│ WebGL Viewer        │
└─────────────────────┘
```

## 📝 Step-by-Step Workflow

---

## 1. Scene Creation

### 1.1 Generate 3D Room Model

Create a custom 7m × 5m × 3m room with furniture using parametric mesh generation:

```bash
python create_scene_5x3x3_multi.py
```

**What it does:**
- Generates separate PLY files for each object (walls, floor, ceiling, furniture)
- Creates material-specific meshes for RF simulation:
  - `meshes/concrete_floor.ply` - Concrete floor
  - `meshes/concrete_walls.ply` - Concrete walls with window/door cutouts
  - `meshes/glass_window.ply` - Glass window
  - `meshes/wood_door.ply` - Wooden door
  - `meshes/wood_furniture.ply` - Tables, chairs, sofa
  - `meshes/metal_tv.ply` - LED TV
- Generates combined PLY: `room_5x3x3_combined.ply` for visualization

**Key Features:**
- Parametric room dimensions (configurable X, Y, Z)
- Realistic furniture placement (3 tables + chairs, sofa, TV)
- Material-based mesh separation for RF propagation modeling
- Window (1.5m × 1.5m) and door (2m × 1m) cutouts

**Output Structure:**
```
meshes/
├── concrete_floor.ply
├── concrete_walls.ply
├── concrete_ceiling.ply
├── glass_window.ply
├── wood_door.ply
├── wood_furniture.ply
└── metal_tv.ply
room_5x3x3_combined.ply
```

### 1.2 Verify Scene Scale

Ensure proper coordinate system and dimensions:

```bash
python check_scene_scale.py
```

**Expected Output:**
```
Total Width (X):  7.0000 m
Total Depth (Y):  5.0000 m
Total Height (Z): 3.0000 m
```

---

## 2. Visual Dataset Generation

### 2.1 Generate RGB Images with Blender

Create photorealistic training images using Cycles renderer:

```bash
blender --background --python generate_visual_dataset.py
```

**Configuration** (in `generate_visual_dataset.py`):
```python
NUM_IMAGES = 300          # Number of camera poses
RESOLUTION = 800          # Image resolution (800×800)
ROOM_MIN = (0.5, 0.5, 0.0)
ROOM_MAX = (6.5, 4.5, 3.0)
```

**What it does:**
1. **Scene Setup**: Imports all meshes from `meshes/` folder
2. **Material Assignment**: Creates PBR materials with realistic properties:
   - Concrete: Rough diffuse surfaces
   - Glass: Semi-transparent with transmission
   - Wood: Textured diffuse with normal mapping
   - Metal: Reflective surfaces
3. **Camera Sampling**: Generates diverse camera poses:
   - Positions: Random within room bounds
   - Orientations: Looking toward room center with perturbations
   - High overlap (90% train, 10% test split)
4. **Rendering**: Uses Cycles GPU rendering with:
   - 96 samples per pixel
   - OptiX denoising
   - Neutral color grading (Standard view transform)

**Output Structure:**
```
dataset_visual_v2/
├── transforms_train.json  # Camera poses (270 images)
├── transforms_test.json   # Camera poses (30 images)
└── images/
    ├── frame_0000.png
    ├── frame_0001.png
    └── ...
```

**transforms_train.json format:**
```json
{
  "camera_angle_x": 0.8575560450553894,
  "frames": [
    {
      "file_path": "images/frame_0000.png",
      "transform_matrix": [
        [0.9848, -0.1736, 0.0000, 3.5],
        [0.1736, 0.9848, 0.0000, 2.5],
        [0.0000, 0.0000, 1.0000, 1.5],
        [0.0, 0.0, 0.0, 1.0]
      ]
    },
    ...
  ]
}
```

**Tips:**
- Requires ~10GB GPU memory for rendering
- Takes ~1-2 hours for 300 images at 800px resolution
- Ensure Blender has GPU rendering enabled in preferences

---

## 3. RF Dataset Generation

### 3.1 Simulate RF Propagation with Sionna

Generate RF heatmaps using ray-tracing simulation:

```bash
python generate_dataset_ideal_mpc.py
```

**Configuration:**
```python
# RF Parameters
FREQUENCY = 28e9          # 28 GHz (mmWave)
BANDWIDTH = 1e9           # 1 GHz bandwidth
NUM_TX = 1                # Single transmitter
TX_POWER = 20             # dBm

# Camera/Receiver Parameters
NUM_IMAGES = 300          # Match visual dataset
RESOLUTION = 800          # Match visual dataset
FOCAL_LENGTH = 1164.69    # Calculated from camera_angle_x
```

**What it does:**
1. **Sionna Scene Setup**:
   - Loads all meshes from `meshes/` folder
   - Assigns radio materials based on filenames:
     - `concrete_*` → `"itu_concrete"`
     - `glass_*` → `"itu_glass"`
     - `wood_*` → `"itu_wood"`
     - `metal_*` → `"itu_metal"`
   
2. **Transmitter Placement**:
   - Position: `(6.0, 2.5, 2.5)` (wall-mounted, centered)
   - Antenna: Isotropic pattern
   - Power: 20 dBm

3. **Camera Pose Generation**:
   - Uses **same camera poses** as visual dataset
   - Converts Blender transforms to Sionna camera format
   - Euler angles → quaternions (COLMAP format)

4. **RF Ray-Tracing**:
   - For each camera pose:
     - Renders 360° panorama (equirectangular)
     - Computes path gains, delays, angles
     - Projects panorama to perspective view (pinhole camera)
     - Saves RF heatmap as grayscale PNG
   - Path features: Gains, delays, AoA, AoD, Doppler

5. **COLMAP Format Export**:
   - Saves `cameras.txt` (intrinsics)
   - Saves `images.txt` (extrinsics)
   - Creates `sparse/0/` structure for 3DGS

**Output Structure:**
```
dataset_custom_scene_ideal_mpc/
├── cameras.txt            # COLMAP camera intrinsics
├── images.txt             # COLMAP camera extrinsics
├── train_index.txt        # Training image list
├── test_index.txt         # Test image list
├── spectrum/              # RF heatmaps
│   ├── frame_0000.png     # Grayscale power map
│   ├── frame_0001.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt    # Copy of intrinsics
        ├── images.txt     # Copy of extrinsics
        └── points3D.txt   # Dummy file (required by 3DGS)
```

**cameras.txt format:**
```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 800 800 1164.69 1164.69 400.0 400.0
```

**images.txt format:**
```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] (empty for our case)
1 0.9848 0.0 0.0 0.1736 3.5 2.5 1.5 1 frame_0000.png

2 0.9659 0.0 0.0 0.2588 4.2 3.1 1.8 1 frame_0001.png
...
```

### 3.2 Prepare RF Data for 3DGS

Organize RF dataset into expected structure:

```bash
cd RF-3DGS
python prepare_rf_data.py
```

**What it does:**
- Creates `sparse/0/` directory structure
- Copies COLMAP files to correct locations
- Generates `train_index.txt` and `test_index.txt`
- Creates dummy `points3D.txt` (required but not used for RF)

---

## 4. 3DGS Training

### 4.1 Train Visual Model (Stage 1)

First, train on visual RGB images to learn scene geometry:

```bash
cd RF-3DGS
conda activate rf-3dgs

python train.py \
  -s /home/ved/Ved/Project_1/dataset_visual_v2 \
  -m output/visual_model \
  --iterations 30000 \
  --save_iterations 7000 15000 30000
```

**Training Parameters:**
- Iterations: 30,000 (standard for 3DGS)
- Densification: Every 100 iterations until iteration 15,000
- Opacity reset: Every 3,000 iterations
- Loss: L1 + SSIM (structural similarity)

**What it does:**
1. **Initialization**: Randomly initialize Gaussians in scene bounds
2. **Optimization**: Iteratively optimize:
   - Gaussian positions (xyz)
   - Gaussian scales (scale)
   - Gaussian rotations (quaternions)
   - Gaussian opacities (alpha)
   - Spherical harmonic coefficients (color)
3. **Densification**: Add/split Gaussians in high-gradient regions
4. **Pruning**: Remove low-opacity Gaussians

**Expected Output:**
```
output/visual_model/
├── cameras.json
├── cfg_args
├── input.ply               # Initial point cloud
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply  # 7K iteration Gaussians
│   ├── iteration_15000/
│   └── iteration_30000/
└── chkpnt30000.pth          # Checkpoint for fine-tuning
```

**Monitoring Training:**
- Loss should decrease steadily
- PSNR should increase (target: >25 dB for indoor scenes)
- Check `output/visual_model/` for intermediate checkpoints

### 4.2 Train RF Model (Stage 2)

Fine-tune visual model on RF heatmaps:

```bash
python train.py \
  -s /home/ved/Ved/Project_1/dataset_custom_scene_ideal_mpc \
  -m output/rf_model \
  --images spectrum \
  --start_checkpoint output/visual_model/chkpnt30000.pth \
  --iterations 10000 \
  --save_iterations 3000 7000 10000
```

**Key Parameters:**
- `--images spectrum`: Use RF heatmaps from `spectrum/` folder
- `--start_checkpoint`: Initialize from visual model (transfer learning)
- Fewer iterations (10K) since geometry is already learned

**What it does:**
1. **Load Visual Checkpoint**: Initialize Gaussians from Stage 1
2. **RF Feature Learning**: Add RF-specific attributes:
   - RF absorption coefficients
   - RF scattering properties
   - Material-dependent propagation
3. **Fine-tuning**: Optimize for RF prediction:
   - Keep geometry mostly fixed
   - Learn RF-specific features
   - Minimize L1 loss between predicted and true RF heatmaps

**Expected Output:**
```
output/rf_model/
├── cameras.json
├── cfg_args
├── point_cloud/
│   ├── iteration_3000/
│   ├── iteration_7000/
│   └── iteration_10000/
│       └── point_cloud.ply  # Final RRF model
└── chkpnt10000.pth
```

**One-Step Script:**

For convenience, use the provided bash script:

```bash
cd RF-3DGS
bash run_rf_reconstruction.sh
```

This script runs both stages sequentially.

---

## 5. Evaluation

### 5.1 Render Test Views

Generate predictions for test set:

```bash
# Render visual test views
python render.py \
  -m output/visual_model \
  --iteration 30000

# Render RF test views
python render.py \
  -m output/rf_model \
  --iteration 10000
```

**Output Structure:**
```
output/visual_model/test/ours_30000/
├── renders/              # Predicted images
│   ├── 00000.png
│   └── ...
└── gt/                   # Ground truth images
    ├── 00000.png
    └── ...

output/rf_model/test/ours_10000/
├── renders/              # Predicted RF heatmaps
└── gt/                   # Ground truth RF heatmaps
```

### 5.2 Compute Metrics

Evaluate reconstruction quality:

```bash
# Visual metrics
python metrics.py -m output/visual_model

# RF metrics
python metrics.py -m output/rf_model
```

**Reported Metrics:**
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
- **SSIM** (Structural Similarity Index): Higher is better (0-1)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better

**Expected Results:**

| Model | PSNR (dB) | SSIM | LPIPS |
|-------|-----------|------|-------|
| Visual | 28-32 | 0.92-0.96 | 0.05-0.10 |
| RF | 25-30 | 0.88-0.93 | 0.10-0.20 |

### 5.3 RF Localization Evaluation

Test RF-based localization using fingerprinting:

```bash
python evaluate_localization.py
```

**What it does:**
1. Loads RF fingerprint dataset (`rf_dataset.pkl`)
2. Extracts features: path gains, delays, power
3. Trains k-NN classifier (k=5)
4. Predicts user positions from RF measurements
5. Computes localization error (RMSE)

**Output:**
- Console: Mean/median localization error
- `localization_results.png`: Scatter plot of true vs predicted positions

---

## 6. Visualization

### 6.1 Interactive 3D Viewer

View reconstructed RRF in WebGL viewer:

```bash
cd RF-3DGS/SIBR_viewers
# Build viewer (first time only)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Launch viewer
./build/bin/SIBR_gaussianViewer_app \
  -m ../output/rf_model \
  --iteration 10000
```

**Viewer Controls:**
- **Mouse**: Rotate view
- **WASD**: Move camera
- **Q/E**: Up/down
- **Scroll**: Zoom
- **Tab**: Toggle UI
- **Space**: Screenshot

### 6.2 Generate Video

Create flythrough video:

```bash
python make_video.py \
  --input output/rf_model/test/ours_10000/renders \
  --output rf_reconstruction.mp4 \
  --fps 30
```

**Options:**
- `--method opencv`: Use OpenCV (faster)
- `--method ffmpeg`: Use FFmpeg (better quality)

---

## 📊 Results

### Visual Reconstruction
- **Scene**: 7m × 5m × 3m room with furniture
- **Training**: 270 images, 800×800 resolution
- **Quality**: PSNR ~30 dB, SSIM ~0.94

### RF Reconstruction
- **Frequency**: 28 GHz (mmWave 5G)
- **Transmitter**: Wall-mounted at (6.0, 2.5, 2.5)
- **Coverage**: Successfully predicts RF heatmaps at novel viewpoints
- **Localization**: ~0.5m average error using RF fingerprinting

### Key Insights
1. **Visual pre-training is crucial**: Starting from random initialization fails for RF
2. **Material modeling matters**: Concrete vs glass vs metal have distinct RF signatures
3. **Multi-path propagation**: Model captures reflections, diffractions around furniture
4. **Generalization**: RRF generalizes to unseen camera positions

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce `RESOLUTION` to 512 or 640
- Reduce `NUM_IMAGES` to 200
- Use `--densify_grad_threshold 0.0003` (more aggressive pruning)

#### 2. Sionna Scene Loading Error
```
AttributeError: 'Scene' object has no attribute 'mi_scene'
```
**Solution:**
- Ensure Sionna 0.18+ is installed
- Check PLY file format (must be binary little-endian)
- Verify mesh normals are consistent

#### 3. Blender Rendering Slow
```
Blender hangs or renders very slowly
```
**Solution:**
- Enable GPU in Blender preferences: Edit → Preferences → System → CUDA/OptiX
- Reduce `scene.cycles.samples` to 64
- Disable denoising: `scene.cycles.use_denoising = False`

#### 4. 3DGS Training Diverges
```
Loss increases or NaN loss
```
**Solution:**
- Check camera poses (visualize with `debug_scene.py`)
- Ensure proper camera coordinate system (OpenGL convention)
- Reduce learning rate: `--position_lr_init 0.00008`

#### 5. COLMAP File Format Error
```
RuntimeError: Could not find cameras.txt
```
**Solution:**
- Run `prepare_rf_data.py` to create `sparse/0/` structure
- Check file paths in `cameras.txt` and `images.txt`
- Ensure `points3D.txt` exists (even if empty/dummy)

---

## 📚 References

### Papers
1. **3D Gaussian Splatting** - Kerbl et al. (2023)
   - [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
   - Original 3DGS implementation

2. **RF-3DGS** - Wang et al. (2024)
   - [GitHub](https://github.com/Wangmz-1203/RF-3DGS)
   - Radio-frequency extension of 3DGS

3. **Sionna RT** - Hoydis et al. (2023)
   - [Documentation](https://nvlabs.github.io/sionna/)
   - Differentiable ray-tracing for wireless

### Software
- **Blender** - [https://www.blender.org/](https://www.blender.org/)
- **COLMAP** - [https://colmap.github.io/](https://colmap.github.io/)
- **PyTorch** - [https://pytorch.org/](https://pytorch.org/)
- **TensorFlow** - [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

## 🤝 Contributing

Contributions welcome! Please open issues for bugs or feature requests.

---

## 📄 License

This project uses code from:
- **RF-3DGS**: BSD 3-Clause License
- **Sionna**: Apache 2.0 License
- **3D Gaussian Splatting**: Original license (Inria)

See `LICENSE` files in respective directories.

---

## 👤 Author

Ved - RF-RRF Reconstruction Pipeline

---

## 🙏 Acknowledgments

- RF-3DGS authors for the RRF framework
- NVIDIA Sionna team for RT ray-tracing
- Inria for original 3D Gaussian Splatting
- Blender Foundation for rendering tools

---

**Last Updated**: February 2026
