# Evaluation and Visualization

Complete guide for evaluating RRF reconstruction quality and visualizing results.

## Overview

After training, you need to:
1. **Render test views** from the trained model
2. **Compute quantitative metrics** (PSNR, SSIM, LPIPS)
3. **Evaluate RF localization** accuracy
4. **Visualize in interactive 3D viewer**
5. **Generate videos and figures**

---

## 1. Rendering Test Views

### Visual Model Rendering

```bash
cd RF-3DGS

# Render test views at iteration 30,000
python render.py \
  -m output/visual_model \
  --iteration 30000 \
  --skip_train \
  --skip_test
```

### RF Model Rendering

```bash
# Render RF heatmaps at iteration 10,000
python render.py \
  -m output/rf_model \
  --iteration 10000 \
  --skip_train
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `-m, --model_path` | Path to trained model |
| `--iteration` | Checkpoint iteration to load |
| `--skip_train` | Don't render training views |
| `--skip_test` | Don't render test views |
| `--skip_video` | Don't generate video |

### Output Structure

```
output/visual_model/
└── test/
    └── ours_30000/
        ├── renders/           # Predicted images
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        └── gt/                # Ground truth images
            ├── 00000.png
            ├── 00001.png
            └── ...

output/rf_model/
└── test/
    └── ours_10000/
        ├── renders/           # Predicted RF heatmaps
        └── gt/                # Ground truth RF heatmaps
```

---

## 2. Quantitative Metrics

### Compute Metrics

```bash
# Visual model metrics
python metrics.py -m output/visual_model

# RF model metrics
python metrics.py -m output/rf_model
```

### Reported Metrics

**1. PSNR (Peak Signal-to-Noise Ratio)**
- Measures pixel-wise reconstruction accuracy
- Higher is better
- Unit: dB (decibels)
- Formula: $\text{PSNR} = 10 \log_{10} \frac{255^2}{\text{MSE}}$

**Interpretation**:
- **> 30 dB**: Excellent quality (visual)
- **25-30 dB**: Good quality
- **20-25 dB**: Acceptable quality (RF)
- **< 20 dB**: Poor quality

**2. SSIM (Structural Similarity Index)**
- Measures perceptual similarity
- Range: [0, 1], higher is better
- Formula: $\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$

**Interpretation**:
- **> 0.95**: Excellent (visual)
- **0.90-0.95**: Good
- **0.85-0.90**: Acceptable (RF)
- **< 0.85**: Poor

**3. LPIPS (Learned Perceptual Image Patch Similarity)**
- Measures perceptual similarity using deep features
- Lower is better
- Range: [0, 1]

**Interpretation**:
- **< 0.10**: Excellent (visual)
- **0.10-0.20**: Good
- **0.20-0.30**: Acceptable (RF)
- **> 0.30**: Poor

### Expected Results

**Visual Model**:
```
Test Set Results:
  PSNR:  30.45 ± 2.13 dB
  SSIM:  0.943 ± 0.021
  LPIPS: 0.082 ± 0.015

Per-image breakdown:
  Image 0: PSNR 31.2 dB, SSIM 0.952, LPIPS 0.073
  Image 1: PSNR 29.8 dB, SSIM 0.938, LPIPS 0.091
  ...
```

**RF Model**:
```
Test Set Results:
  PSNR:  27.34 ± 1.87 dB
  SSIM:  0.891 ± 0.032
  LPIPS: 0.187 ± 0.042

Per-image breakdown:
  Image 0: PSNR 28.1 dB, SSIM 0.902, LPIPS 0.165
  Image 1: PSNR 26.7 dB, SSIM 0.883, LPIPS 0.203
  ...
```

### Save Metrics to File

```bash
# Save metrics as JSON
python metrics.py -m output/visual_model > visual_metrics.json
python metrics.py -m output/rf_model > rf_metrics.json
```

---

## 3. RF Localization Evaluation

### Purpose

Test the utility of RRF for RF-based indoor localization using k-NN fingerprinting.

### Generate Localization Dataset

First, save RF path features during dataset generation:

```python
# In generate_dataset_ideal_mpc.py
import pickle

dataset = []
for i, pose in enumerate(camera_poses):
    # Compute paths
    paths = scene.compute_paths()
    
    # Extract features
    features = {
        'path_gains': np.abs(paths.a.numpy()).flatten(),
        'path_delays': paths.tau.numpy().flatten(),
        'total_power': np.sum(np.abs(paths.a.numpy())**2),
        'num_paths': len(paths.a.numpy().flatten()),
        'delay_spread': np.std(paths.tau.numpy())
    }
    
    dataset.append({
        'position': pose[:3, 3],
        'features': features,
        'image_id': i
    })

# Save dataset
with open('rf_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

### Run Localization Evaluation

```bash
python evaluate_localization.py
```

### Algorithm

**k-Nearest Neighbors (k-NN) Localization**:

1. **Split Dataset**: 80% train, 20% test
2. **Feature Extraction**: Top-10 path gains + delays
3. **Distance Metric**: Euclidean distance in feature space
4. **Position Estimation**: Weighted average of k nearest neighbors
5. **Evaluation**: Compute RMSE between true and predicted positions

### Output

**Console**:
```
Loading dataset from: rf_dataset.pkl
✓ Loaded 300 samples

Preparing data...
  Training samples: 240
  Test samples: 60
  Feature dimension: 23

Training k-NN model (k=5)...
✓ Model trained

Evaluating localization...
Progress: [========================================] 60/60

===== Localization Results =====
Mean Error:   0.487 m
Median Error: 0.421 m
90th percentile: 0.823 m
Max Error:    1.234 m

RMSE: 0.542 m

✓ Saved results to localization_results.png
```

**Visualization**: `localization_results.png`
- Scatter plot: True positions (blue) vs Predicted positions (red)
- Error vectors: Lines connecting true and predicted positions
- Colormap: Error magnitude

### Advanced: Neural Network Localization

Train a deep neural network for better accuracy:

```bash
python train_nn_localizer.py
```

**Architecture**:
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(feature_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3)  # Output: (x, y, z)
])
```

**Expected Improvement**:
- k-NN: ~0.5m RMSE
- Neural Network: ~0.3m RMSE

---

## 4. Interactive 3D Viewer

### Build SIBR Viewer (First Time Only)

```bash
cd RF-3DGS/SIBR_viewers

# Install dependencies
sudo apt-get install -y \
    cmake \
    libglew-dev \
    libglfw3-dev \
    libboost-all-dev \
    libassimp-dev \
    libopencv-dev

# Build viewer
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Launch Viewer

**Visual Model**:
```bash
cd RF-3DGS/SIBR_viewers
./build/bin/SIBR_gaussianViewer_app \
  -m ../output/visual_model \
  --iteration 30000
```

**RF Model**:
```bash
./build/bin/SIBR_gaussianViewer_app \
  -m ../output/rf_model \
  --iteration 10000
```

### Viewer Controls

**Camera**:
- **Left Mouse**: Rotate view
- **Right Mouse**: Pan
- **Scroll**: Zoom in/out
- **W/A/S/D**: Move camera forward/left/back/right
- **Q/E**: Move camera down/up
- **Shift**: Move faster

**Display**:
- **Tab**: Toggle UI
- **Space**: Take screenshot
- **R**: Reset camera
- **1-9**: Switch between training views

**Rendering**:
- **F**: Toggle FPS display
- **G**: Toggle ground plane
- **B**: Toggle bounding box
- **N**: Toggle normals

### Screenshot

Press **Space** to save screenshot to:
```
RF-3DGS/SIBR_viewers/screenshots/screenshot_YYYYMMDD_HHMMSS.png
```

### Web-based Viewer

For WebGL viewer (no build required):

```bash
# Convert model to web format
python scripts/export_webgl.py \
  -m output/rf_model \
  --iteration 10000 \
  -o webgl_export/

# Start local server
cd webgl_export
python -m http.server 8000

# Open browser: http://localhost:8000
```

---

## 5. Video Generation

### Render Trajectory Video

**Using render.py**:
```bash
python render.py \
  -m output/rf_model \
  --iteration 10000 \
  --render_path
```

Generates circular camera path video.

**Custom Trajectory**:
```bash
python make_video.py \
  --input output/rf_model/test/ours_10000/renders \
  --output rf_reconstruction.mp4 \
  --fps 30
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--input` | Directory with frame images |
| `--output` | Output video path (.mp4) |
| `--fps` | Frames per second (default: 30) |
| `--method` | Encoding method: `opencv` or `ffmpeg` |

### Methods

**OpenCV** (faster, lower quality):
```bash
python make_video.py \
  --input output/rf_model/test/ours_10000/renders \
  --output rf_video.mp4 \
  --method opencv
```

**FFmpeg** (slower, better quality):
```bash
python make_video.py \
  --input output/rf_model/test/ours_10000/renders \
  --output rf_video_hq.mp4 \
  --method ffmpeg
```

### Advanced: Side-by-Side Comparison

```bash
# Create comparison video
ffmpeg -i visual_%04d.png -i rf_%04d.png \
  -filter_complex "[0:v][1:v]hstack=inputs=2" \
  -c:v libx264 -pix_fmt yuv420p \
  comparison.mp4
```

---

## 6. Qualitative Analysis

### Visual Inspection

**Checklist**:
- [ ] Rendered images are sharp (no blur)
- [ ] Colors match ground truth
- [ ] Furniture geometry is correct (no holes/artifacts)
- [ ] Window and door are properly rendered (transparency)
- [ ] No floating Gaussians (artifacts)
- [ ] Shadows and lighting look realistic

**Common Issues**:
- **Blurry images**: Insufficient training iterations
- **Holes in surfaces**: Too aggressive pruning
- **Floating artifacts**: Under-regularized (need more views)
- **Color mismatch**: Lighting inconsistency in dataset

### RF Inspection

**Checklist**:
- [ ] RF heatmaps show gradient from TX to corners
- [ ] Stronger signal near transmitter position (6.0, 2.5, 2.5)
- [ ] Weaker signal in shadowed regions (behind furniture)
- [ ] Reflections visible (bright spots from walls)
- [ ] No completely black images (signal everywhere)

**Common Issues**:
- **Uniform heatmaps**: Model didn't learn RF patterns (need more iterations)
- **Black spots**: Over-regularized (geometry too rigid)
- **Unrealistic patterns**: Dataset quality issue (check Sionna simulation)

---

## 7. Comparative Analysis

### Baseline Comparison

**k-NN Interpolation Baseline**:
```python
# Simple baseline: interpolate from nearest training view
from scipy.spatial import KDTree

# Build tree from training positions
tree = KDTree(train_positions)

# For each test position, find nearest training view
for test_pos in test_positions:
    dist, idx = tree.query(test_pos, k=1)
    baseline_prediction = train_images[idx]
    # Compare with 3DGS prediction
```

**Expected Comparison**:
| Method | PSNR (dB) | SSIM | LPIPS |
|--------|-----------|------|-------|
| Nearest Neighbor | 22.3 | 0.812 | 0.312 |
| **3DGS (Ours)** | **30.5** | **0.943** | **0.082** |

### Ablation Study

Test importance of visual pre-training:

**Without Pre-training** (train RF from scratch):
```bash
python train.py \
  -s dataset_custom_scene_ideal_mpc \
  -m output/rf_from_scratch \
  --images spectrum \
  --iterations 30000
```

**Expected Result**: Much lower quality (PSNR ~18 dB)

**With Pre-training** (our method):
```bash
# Stage 1 + Stage 2 as before
```

**Expected Result**: High quality (PSNR ~27 dB)

**Conclusion**: Visual pre-training is crucial for learning RF patterns.

---

## 8. Export Results

### Generate Report

Create a comprehensive results report:

```bash
python scripts/summary.py \
  --visual_model output/visual_model \
  --rf_model output/rf_model \
  --output results_report.pdf
```

**Report Contents**:
1. Dataset statistics (# images, resolution, coverage)
2. Training curves (loss, PSNR over iterations)
3. Quantitative metrics (PSNR, SSIM, LPIPS)
4. Qualitative results (rendered images, RF heatmaps)
5. Localization accuracy (RMSE, error distribution)
6. Ablation studies (with/without pre-training)

### Export for Publication

**High-resolution renders**:
```bash
python render.py \
  -m output/rf_model \
  --iteration 10000 \
  --resolution 2048  # 2K resolution
```

**Figures for paper**:
```python
import matplotlib.pyplot as plt

# Plot RF heatmaps
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    img = plt.imread(f'renders/{i:05d}.png')
    ax.imshow(img, cmap='jet')
    ax.set_title(f'View {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('rf_heatmaps.pdf', dpi=300, bbox_inches='tight')
```

---

## 9. Troubleshooting

### Low PSNR on Test Set

**Symptom**: PSNR < 25 dB on visual or RF test set

**Diagnosis**:
```bash
# Check training set performance
python render.py -m output/visual_model --skip_test
python metrics.py -m output/visual_model
```

**If training PSNR is also low**: Underfitting
- Solution: Train longer, increase capacity

**If training PSNR is high**: Overfitting
- Solution: More training views, data augmentation

### Artifacts in Renders

**Symptom**: Floating blobs, holes, or distortions

**Causes**:
1. Insufficient camera coverage
2. Over-pruning during training
3. Incorrect camera poses

**Solutions**:
- Increase densification threshold: `--densify_grad_threshold 0.0001`
- Check camera poses: `python debug_scene.py`
- Add more training views

### Slow Rendering

**Symptom**: Viewer runs at <30 FPS

**Solutions**:
- Reduce Gaussian count: More aggressive pruning
- Lower rendering resolution in viewer
- Enable culling: Only render visible Gaussians

---

## 10. Best Practices

### Evaluation Checklist

- [ ] Render both train and test views
- [ ] Compute all metrics (PSNR, SSIM, LPIPS)
- [ ] Evaluate RF localization accuracy
- [ ] Visualize in interactive viewer
- [ ] Generate videos for qualitative assessment
- [ ] Compare with baselines
- [ ] Perform ablation studies
- [ ] Export high-resolution results

### Reproducibility

To ensure reproducible results:
1. **Fix random seeds**: 
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```
2. **Document hyperparameters**: Save `cfg_args`
3. **Version control**: Track code and config changes
4. **Log training**: Use TensorBoard or Weights & Biases

### Reporting Guidelines

When reporting results:
- **Metrics**: Report mean ± std over test set
- **Timing**: Report training time and hardware specs
- **Comparisons**: Compare with published baselines
- **Limitations**: Discuss failure cases and limitations

---

## Next Steps

After evaluation:
1. **Analyze results**: Identify strengths and weaknesses
2. **Iterate**: Improve dataset or training if needed
3. **Document**: Write up findings in report/paper
4. **Share**: Push code and results to GitHub

---

**See also**:
- [Main README](../README.md) - Complete pipeline
- [Training Guide](TRAINING.md) - Improve model quality
- [Troubleshooting](../README.md#troubleshooting) - Common issues

---

**Quality Benchmarks**:

| Model | PSNR (dB) | SSIM | LPIPS | Training Time |
|-------|-----------|------|-------|---------------|
| Visual 3DGS | 30-32 | 0.92-0.96 | 0.05-0.10 | 2-4 hours |
| RF-RRF | 25-30 | 0.88-0.93 | 0.10-0.20 | 1-2 hours |
| RF Localization | 0.3-0.5m RMSE | - | - | - |

---

**Evaluation Complete!** 🎉

You now have comprehensive metrics and visualizations of your RRF reconstruction.
