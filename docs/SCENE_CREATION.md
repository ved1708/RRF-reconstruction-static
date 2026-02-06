# 3D Scene Creation Scripts

This directory contains scripts for creating custom 3D room models for RF propagation simulation and 3D Gaussian Splatting.

## Scripts Overview

### 1. create_scene_5x3x3_multi.py

**Purpose**: Generate parametric 3D room model with furniture as separate material-based PLY files.

**Features**:
- Creates 7m × 5m × 3m rectangular room
- Separate meshes for each material type (concrete, glass, wood, metal)
- Realistic furniture placement (tables, chairs, sofa, TV)
- Window and door cutouts in walls
- Combined PLY file for visualization

**Usage**:
```bash
python create_scene_5x3x3_multi.py
```

**Output**:
```
meshes/
├── concrete_floor.ply    - Floor surface (7m × 5m)
├── concrete_walls.ply    - Walls with cutouts (3m height)
├── concrete_ceiling.ply  - Ceiling surface
├── glass_window.ply      - Window (1.5m × 1.5m)
├── wood_door.ply         - Door (2m × 1m)
├── wood_furniture.ply    - Tables + chairs + sofa
└── metal_tv.ply          - LED TV (0.05m × 1.2m × 0.7m)
room_5x3x3_combined.ply   - All objects combined
```

**Room Layout**:
```
     5m (Y-axis)
    ┌────────────────┐
    │                │
    │   Table 1      │
 7m │      □         │ Window (Y=5m wall)
(X) │                │ at (3.5, 5.0, 1.5)
    │   Table 2      │
    │      □         │
    │                │
    │   Table 3  TV  │
    │      □    ▭    │
    │                │
    │   Sofa         │ Door (Y=0 wall)
    │   ▬▬▬▬         │ at (1.0, 0.0, 1.0)
    └────────────────┘
    0.5  →  6.5 (X-axis bounds)
```

**Furniture Details**:
- **Tables**: 3 tables (1.2m × 0.6m × 0.75m) with 4 chairs each
- **Chairs**: 0.4m × 0.4m × 0.45m seats, 0.9m backrest height
- **Sofa**: 2m × 0.8m × 0.4m at position (3.5, 1.5, 0.4)
- **TV**: 1.2m × 0.7m screen at position (6.5, 2.5, 2.0)

**Material Properties** (for Sionna RT):
| Material | ITU Code | Usage |
|----------|----------|-------|
| Concrete | `itu_concrete` | Floor, walls, ceiling |
| Glass | `itu_glass` | Window |
| Wood | `itu_wood` | Door, furniture |
| Metal | `itu_metal` | TV frame |

**Coordinate System**:
- Origin: (0, 0, 0) at corner
- X-axis: Room width (7m)
- Y-axis: Room depth (5m)
- Z-axis: Height (3m)
- Units: Meters

**Customization**:
Edit these variables in the script:
```python
ROOM_WIDTH = 7.0    # X-dimension
ROOM_DEPTH = 5.0    # Y-dimension
ROOM_HEIGHT = 3.0   # Z-dimension
WALL_THICKNESS = 0.2

WINDOW_POS = (3.5, 5.0, 1.5)  # Center position
WINDOW_SIZE = (1.5, 1.5)      # Width × Height

DOOR_POS = (1.0, 0.0, 0.0)    # Bottom-left corner
DOOR_SIZE = (1.0, 2.0)        # Width × Height
```

---

### 2. create_scene.py

**Purpose**: Simplified version creating basic room without furniture.

**Usage**:
```bash
python create_scene.py
```

**Output**: Single combined PLY file with walls, floor, ceiling.

---

### 3. check_scene_scale.py

**Purpose**: Verify scene dimensions and coordinate system before RF simulation.

**Usage**:
```bash
python check_scene_scale.py
```

**What it checks**:
1. Loads scene XML file (for Sionna)
2. Extracts bounding box from Mitsuba scene
3. Computes dimensions (width, depth, height)
4. Verifies scale is realistic (meters, not millimeters)

**Expected Output**:
```
--- Checking Scene Scale: room_5x3x3.xml ---
============================================================
METRIC               | VALUE (Meters)
------------------------------------------------------------
Min Coordinates      | [0.5 0.5 0.0]
Max Coordinates      | [6.5 4.5 3.0]
------------------------------------------------------------
Total Width (X)      | 7.0000 m
Total Depth (Y)      | 5.0000 m
Total Height (Z)     | 3.0000 m
============================================================

--- DIAGNOSIS ---
✓ Scene dimensions are realistic for an indoor room.
✓ Coordinate system uses meters (standard for Sionna RT).
```

**Common Issues**:
- **Scale too small** (<0.1m): Scene may be in millimeters → rescale meshes
- **Scale too large** (>100m): Likely coordinate error
- **Negative coordinates**: Check mesh export settings

---

### 4. debug_scene.py

**Purpose**: Visualize camera poses and scene geometry for debugging.

**Usage**:
```bash
python debug_scene.py
```

**Features**:
- Loads COLMAP camera poses
- Visualizes camera frustums in 3D
- Overlays room mesh
- Checks for invalid poses (cameras outside room, etc.)

---

### 5. las2ply.py

**Purpose**: Convert LAS/LAZ point cloud files to PLY format.

**Usage**:
```bash
python las2ply.py input.las output.ply
```

Used for processing real-world LiDAR scans (e.g., ArtPark dataset).

---

## PLY File Format

All generated PLY files use **binary little-endian** format:

```
ply
format binary_little_endian 1.0
element vertex <N>
property float x
property float y
property float z
element face <M>
property list uchar int vertex_indices
property int material_id  # (optional, for combined files)
end_header
<binary vertex data>
<binary face data>
```

**Why binary?**
- Smaller file size
- Faster loading in Sionna/Blender/3DGS
- Required by Mitsuba scene loader

---

## Sionna Scene XML

To use generated meshes in Sionna RT, create an XML file:

**room_5x3x3.xml**:
```xml
<?xml version="1.0" encoding="utf-8"?>
<scene version="3.0.0">
    <!-- Floor -->
    <shape type="ply">
        <string name="filename" value="meshes/concrete_floor.ply"/>
        <ref id="itu_concrete" name="bsdf"/>
    </shape>
    
    <!-- Walls -->
    <shape type="ply">
        <string name="filename" value="meshes/concrete_walls.ply"/>
        <ref id="itu_concrete" name="bsdf"/>
    </shape>
    
    <!-- Ceiling -->
    <shape type="ply">
        <string name="filename" value="meshes/concrete_ceiling.ply"/>
        <ref id="itu_concrete" name="bsdf"/>
    </shape>
    
    <!-- Window -->
    <shape type="ply">
        <string name="filename" value="meshes/glass_window.ply"/>
        <ref id="itu_glass" name="bsdf"/>
    </shape>
    
    <!-- Door -->
    <shape type="ply">
        <string name="filename" value="meshes/wood_door.ply"/>
        <ref id="itu_wood" name="bsdf"/>
    </shape>
    
    <!-- Furniture -->
    <shape type="ply">
        <string name="filename" value="meshes/wood_furniture.ply"/>
        <ref id="itu_wood" name="bsdf"/>
    </shape>
    
    <!-- TV -->
    <shape type="ply">
        <string name="filename" value="meshes/metal_tv.ply"/>
        <ref id="itu_metal" name="bsdf"/>
    </shape>
</scene>
```

Load in Python:
```python
from sionna.rt import load_scene

scene = load_scene("room_5x3x3.xml")
```

---

## Tips

### Mesh Quality
- Ensure **manifold geometry** (no holes, no intersecting faces)
- Use **consistent winding order** (counter-clockwise for outward normals)
- Avoid **degenerate triangles** (zero area)

### Performance
- Keep triangle count reasonable (<100K faces per mesh)
- Merge nearby vertices to avoid duplicates
- Use coarser meshes for large flat surfaces (floor, walls)

### Debugging
1. **Visualize in Blender**:
   ```bash
   blender room_5x3x3_combined.ply
   ```
2. **Check normals**: Edit Mode → Mesh → Normals → Recalculate Outside
3. **Validate**: Edit Mode → Mesh → Clean Up → Merge by Distance

---

## Next Steps

After creating the scene:
1. Generate visual dataset → `../generate_visual_dataset.py`
2. Generate RF dataset → `../generate_dataset_ideal_mpc.py`
3. Train 3DGS → `../RF-3DGS/train.py`

---

**See also**: [Main README](../README.md) for complete pipeline.
