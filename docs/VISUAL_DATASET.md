# Visual Dataset Generation with Blender

Complete guide for generating photorealistic RGB image datasets for 3D Gaussian Splatting training.

## Overview

This process creates a synthetic dataset of RGB images with accurate camera poses using Blender's Cycles renderer. The dataset serves as the foundation for training the visual geometry component of the RRF model.

---

## Script: generate_visual_dataset.py

### Purpose

Generate 300 high-quality RGB images from diverse camera viewpoints inside the 3D room scene, formatted for 3DGS training.

### Requirements

- **Blender 3.6+** (with GPU support recommended)
- **Python modules**: `bpy`, `mathutils` (built into Blender)
- **Input**: PLY meshes from `meshes/` directory
- **Hardware**: 8GB+ GPU VRAM for fast rendering

---

## Configuration

Key parameters in `generate_visual_dataset.py`:

```python
# === PATHS ===
BASE_DIR = "/home/ved/Ved/Project_1"
INPUT_MODELS_DIR = os.path.join(BASE_DIR, "meshes")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_visual_v2")

# === DATASET SIZE ===
NUM_IMAGES = 300          # Total images to generate
RESOLUTION = 800          # Image resolution (square: 800×800)

# === ROOM BOUNDS ===
ROOM_MIN = mathutils.Vector((0.5, 0.5, 0.0))  # Minimum (X, Y, Z)
ROOM_MAX = mathutils.Vector((6.5, 4.5, 3.0))  # Maximum (X, Y, Z)
CENTER = (ROOM_MIN + ROOM_MAX) / 2             # Room center (3.5, 2.5, 1.5)

# === MATERIAL COLORS ===
MATERIAL_COLORS = {
    "walls": (0.85, 0.85, 0.85, 1),      # Light gray
    "floor": (0.25, 0.25, 0.25, 1),      # Dark gray
    "ceiling": (0.95, 0.95, 0.95, 1),    # White
    "door": (0.4, 0.2, 0.1, 1),          # Brown
    "window": (0.7, 0.8, 1.0, 1),        # Transparent blue-tint
    "furniture": (0.55, 0.35, 0.15, 1),  # Wood brown
    "led_tv": (0.05, 0.05, 0.05, 1)      # Black
}
```

---

## Workflow Steps

### 1. Scene Setup

**Import Meshes**:
- Loads all PLY files from `meshes/` directory
- Assigns materials based on filename prefix:
  - `concrete_*` → walls/floor/ceiling material
  - `glass_*` → window material (semi-transparent)
  - `wood_*` → door/furniture material
  - `metal_*` → TV material

**Material Creation**:
- Uses **Principled BSDF** shader node for PBR rendering
- Special handling for glass:
  ```python
  if is_glass:
      bsdf.inputs['Transmission Weight'].default_value = 0.65  # Semi-transparent
      bsdf.inputs['Roughness'].default_value = 0.3            # Slight roughness
      # Add procedural noise for "smudges"
      noise = nodes.new(type='ShaderNodeTexNoise')
      noise.inputs['Scale'].default_value = 50.0
  ```

### 2. Lighting Setup

**HDRI Environment**:
- Loads `interior_HDRI.exr` for realistic ambient lighting
- Fallback: Create artificial area lights if HDRI not found

**Area Lights** (fallback):
```python
# Ceiling light (bright white)
location=(3.5, 2.5, 2.9)
energy=200
size=2.0

# Side lights for fill
location=(1.0, 2.5, 2.0), energy=50
location=(6.0, 2.5, 2.0), energy=50
```

### 3. Camera Pose Sampling

**Strategy**: Random viewpoints with constraints

```python
def sample_camera_position():
    # Random position within room bounds
    x = random.uniform(ROOM_MIN.x, ROOM_MAX.x)
    y = random.uniform(ROOM_MIN.y, ROOM_MAX.y)
    z = random.uniform(0.8, 2.5)  # Eye height range
    
    # Look toward room center with perturbation
    direction = CENTER - position
    direction.normalize()
    
    # Add random rotation (±30°)
    yaw_offset = random.uniform(-30, 30)
    pitch_offset = random.uniform(-15, 15)
    
    return position, rotation
```

**Constraints**:
- Avoid camera positions inside furniture (collision check)
- Maintain minimum distance to walls (0.3m margin)
- Ensure camera looks inward (no views facing walls directly)
- High overlap for SfM: neighboring views share >60% FOV

### 4. Rendering

**Cycles GPU Rendering**:
```python
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.compute_device_type = 'CUDA'  # or 'OPTIX' for RTX

# Quality settings
scene.cycles.samples = 96              # Samples per pixel
scene.cycles.use_denoising = True      # Enable OptiX denoiser
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.max_bounces = 3           # Light bounces

# Resolution
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.film_transparent = False  # White background
```

**Color Management**:
```python
# Neutral tone mapping (avoid color shift)
scene.view_settings.view_transform = 'Standard'
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
```

**Render Time**:
- ~15-30 seconds per image (with RTX 3080)
- Total: 2-3 hours for 300 images

### 5. Output Format

**Directory Structure**:
```
dataset_visual_v2/
├── transforms_train.json  # 90% of images (270)
├── transforms_test.json   # 10% of images (30)
└── images/
    ├── frame_0000.png
    ├── frame_0001.png
    ├── frame_0002.png
    └── ... (300 images total)
```

**transforms_train.json Format**:
```json
{
  "camera_angle_x": 0.8575560450553894,  # Horizontal FOV in radians
  "frames": [
    {
      "file_path": "images/frame_0000.png",
      "transform_matrix": [
        [r11, r12, r13, tx],  # Camera-to-world rotation + translation
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0.0, 0.0, 0.0, 1.0]
      ]
    },
    ...
  ]
}
```

**Coordinate System**:
- **Blender/NeRF convention**: OpenGL-style
  - +X: Right
  - +Y: Up
  - +Z: Backward (camera looks toward -Z)
- Transform matrix: **Camera-to-World** (C2W)

**Camera Intrinsics**:
- **FOV**: 49.13° (horizontal)
- **Focal Length**: Calculated from FOV: `f = width / (2 * tan(fov/2))`
- **Principal Point**: Image center `(cx, cy) = (400, 400)`

---

## Usage

### Basic Execution

```bash
# Run in Blender (headless mode)
blender --background --python generate_visual_dataset.py
```

### With Blender UI (for debugging)

```bash
# Open Blender, then run script in Scripting workspace
blender  # Launch Blender GUI
# Navigate to Scripting tab
# Open generate_visual_dataset.py
# Click "Run Script"
```

### Monitor Progress

The script prints progress:
```
Setting up render engine...
Enabled GPU: NVIDIA GeForce RTX 3080 (CUDA)
Creating materials...
Loaded: meshes/concrete_floor.ply
Loaded: meshes/concrete_walls.ply
...
Rendering frame 0/300...
Rendering frame 1/300...
...
Done! Generated 270 train and 30 test frames
Dataset saved to: /home/ved/Ved/Project_1/dataset_visual_v2
```

---

## Customization

### Increase Image Count
```python
NUM_IMAGES = 500  # More images = better 3DGS quality
```

### Higher Resolution
```python
RESOLUTION = 1024  # Better detail, but slower rendering
```

### Faster Rendering (lower quality)
```python
scene.cycles.samples = 64          # Fewer samples
scene.cycles.use_denoising = False # Disable denoiser
scene.cycles.max_bounces = 2       # Fewer light bounces
```

### Change Train/Test Split
```python
test_ratio = 0.15  # 15% test, 85% train
num_test = int(NUM_IMAGES * test_ratio)
```

---

## Troubleshooting

### Issue 1: Blender Crashes (Out of Memory)

**Symptom**: Blender terminates during rendering

**Solution**:
```python
# Reduce resolution
RESOLUTION = 640

# Enable tile-based rendering
scene.cycles.use_progressive_refine = False
scene.render.tile_x = 256
scene.render.tile_y = 256
```

### Issue 2: Slow Rendering (CPU instead of GPU)

**Symptom**: Each frame takes >5 minutes

**Solution**:
1. Check GPU availability:
   ```python
   prefs = bpy.context.preferences
   cycles_prefs = prefs.addons['cycles'].preferences
   print([d.name for d in cycles_prefs.devices])
   ```
2. Enable GPU in Blender UI: Edit → Preferences → System → CUDA/OptiX
3. Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall`

### Issue 3: Black Images

**Symptom**: Rendered images are completely black

**Possible Causes**:
1. **No lights**: Ensure HDRI or area lights are enabled
2. **Camera inside wall**: Check collision detection
3. **Wrong render engine**: Ensure `scene.render.engine = 'CYCLES'`

**Debug**:
```python
# Manually test render one frame
bpy.ops.render.render(write_still=True)
```

### Issue 4: Incorrect Camera Poses

**Symptom**: 3DGS training fails with "camera pose error"

**Verification**:
```python
# Check transform matrix
T = cam.matrix_world
print("Camera position:", T.translation)
print("Camera rotation:", T.to_euler())
```

**Expected**: Camera should be inside room bounds `(0.5-6.5, 0.5-4.5, 0.0-3.0)`

---

## Advanced Tips

### Add Motion Blur
```python
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.5
```

### Add Depth-of-Field
```python
cam.data.dof.use_dof = True
cam.data.dof.focus_distance = 3.0
cam.data.dof.aperture_fstop = 2.8
```

### Export Depth Maps
```python
# Enable depth output
scene.use_nodes = True
nodes = scene.node_tree.nodes
render_layers = nodes['Render Layers']
depth_output = render_layers.outputs['Depth']
# Save depth as EXR
```

### Use Real HDRI
Download from [Poly Haven](https://polyhaven.com/hdris):
```python
hdri_path = "path/to/hdri/interior_modern_01_4k.exr"
env_texture = nodes.new(type='ShaderNodeTexEnvironment')
env_texture.image = bpy.data.images.load(hdri_path)
```

---

## Quality Checklist

Before proceeding to 3DGS training, verify:

- [ ] 300 images generated successfully
- [ ] `transforms_train.json` and `transforms_test.json` exist
- [ ] Images are sharp, well-lit, no black frames
- [ ] Camera poses are diverse (not all from same corner)
- [ ] Images have high overlap (neighboring views share features)
- [ ] No missing textures or pink materials in renders
- [ ] Resolution matches configuration (800×800)

---

## Next Steps

After generating the visual dataset:
1. **Verify data**: Open a few images to check quality
2. **Generate RF dataset**: `python generate_dataset_ideal_mpc.py`
3. **Train 3DGS visual model**: `python RF-3DGS/train.py -s dataset_visual_v2`

---

**See also**:
- [Main README](../README.md) - Complete pipeline
- [RF Dataset Generation](RF_DATASET.md) - Next step
- [3DGS Training](TRAINING.md) - Model training

---

**Rendering Tips**:
- Use **OptiX denoiser** (faster than OpenImageDenoise)
- Enable **adaptive sampling** for faster convergence
- Save intermediate checkpoints every 50 frames
- Render in batches if memory limited

