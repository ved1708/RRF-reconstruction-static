import bpy
import os
import json
import math
import mathutils
import random
import numpy as np

# ================= CONFIGURATION =================
BASE_DIR = "/home/ved/Ved/Project_1"
INPUT_MODELS_DIR = os.path.join(BASE_DIR, "meshes")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_visual_v2")

# High overlap is key for geometry
NUM_IMAGES = 300 
RESOLUTION = 800  # High resolution for better quality


# Room Dimensions
ROOM_MIN = mathutils.Vector((0.5, 0.5, 0.0))
ROOM_MAX = mathutils.Vector((6.5, 4.5, 3.0)) 
CENTER = (ROOM_MIN + ROOM_MAX) / 2

MATERIAL_COLORS = {
    "walls": (0.85, 0.85, 0.85, 1),
    "floor": (0.25, 0.25, 0.25, 1),
    "ceiling": (0.95, 0.95, 0.95, 1),
    "door": (0.4, 0.2, 0.1, 1),
    "window": (0.7, 0.8, 1.0, 1),
    "furniture": (0.55, 0.35, 0.15, 1),
    "led_tv": (0.05, 0.05, 0.05, 1)
}

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 96  # Higher quality with GPU acceleration
    scene.cycles.use_denoising = True  # Re-enabled for cleaner images
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Fast denoiser
    scene.cycles.max_bounces = 3  # Increased for better lighting
    
    # Enable GPU rendering (CRITICAL for speed)
    scene.cycles.device = 'GPU'
    
    # Get cycles preferences and enable GPU
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        
        # Enable CUDA/OptiX (for NVIDIA GPUs)
        cycles_prefs.refresh_devices()
        cycles_prefs.compute_device_type = 'CUDA'  # Try CUDA first
        
        # Enable all GPU devices
        for device in cycles_prefs.devices:
            if device.type in ('CUDA', 'OPTIX', 'HIP'):
                device.use = True
                print(f"Enabled GPU: {device.name} ({device.type})")
    except Exception as e:
        print(f"GPU setup warning: {e}")
        print("Falling back to CPU rendering")
    
    # ---  Neutral Exposure ---
    if hasattr(scene.view_settings, 'view_transform'):
        scene.view_settings.view_transform = 'Standard' 
    
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = True

def create_high_feature_material(name, color, is_glass=False):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    bsdf.inputs['Base Color'].default_value = color
    
    if is_glass:
        # ===  SCANNER-FRIENDLY GLASS ===
        # 1. Semi-transparent
        bsdf.inputs['Transmission Weight'].default_value = 0.65  
        
        # 2. Roughness for reflections
        bsdf.inputs['Roughness'].default_value = 0.3 
        
        # 3. Add "Smudges" (Noise)
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 50.0 
        links.new(noise.outputs['Fac'], bsdf.inputs['Alpha'])
        
        # IMPORTANT: Only 'blend_method' is needed for viewport transparency
        mat.blend_method = 'BLEND' 
        # mat.shadow_method = 'HASHED'  <-- DELETED (Not needed for Cycles)

    else:
        # Standard Wall/Floor Material
        noise_large = nodes.new(type='ShaderNodeTexNoise')
        noise_large.inputs['Scale'].default_value = 15.0
        
        noise_small = nodes.new(type='ShaderNodeTexNoise')
        noise_small.inputs['Scale'].default_value = 100.0
        
        mix_rgb = nodes.new(type='ShaderNodeMixRGB')
        mix_rgb.blend_type = 'ADD'
        mix_rgb.inputs['Fac'].default_value = 0.5
        links.new(noise_large.outputs['Fac'], mix_rgb.inputs['Color1'])
        links.new(noise_small.outputs['Fac'], mix_rgb.inputs['Color2'])
        
        bump = nodes.new(type='ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.2 
        
        links.new(mix_rgb.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        bsdf.inputs['Roughness'].default_value = 0.8 

    return mat

def setup_lighting():
    scene = bpy.context.scene
    
    if not scene.world:
        new_world = bpy.data.worlds.new("World")
        scene.world = new_world

    # 1. Background (Fill)
    world = scene.world
    world.use_nodes = True
    
    bg = None
    if 'Background' in world.node_tree.nodes:
        bg = world.node_tree.nodes['Background']
    else:
        bg = world.node_tree.nodes.new('ShaderNodeBackground')
        output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])
        
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.6

    # 2. Sun (Key)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.object
    sun.data.energy = 2.5
    sun.data.angle = 0.5 
    sun.rotation_euler = (math.radians(45), math.radians(15), 0)

    # 3. Ceiling Panel
    bpy.ops.object.light_add(type='AREA', location=(3.5, 2.5, 2.9)) 
    ceiling_light = bpy.context.object
    ceiling_light.name = "Ceiling_Panel"
    ceiling_light.data.energy = 250.0
    ceiling_light.data.size = 4.0     
    ceiling_light.data.color = (1.0, 0.98, 0.9) # RGB only

def import_models():
    if not os.path.exists(INPUT_MODELS_DIR): return
    files = [f for f in os.listdir(INPUT_MODELS_DIR) if f.endswith('.ply')]
    
    for filename in files:
        full_path = os.path.join(INPUT_MODELS_DIR, filename)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.wm.ply_import(filepath=full_path)
        
        if not bpy.context.selected_objects: continue
        obj = bpy.context.selected_objects[0]
        
        # Determine Material
        color = (0.5, 0.5, 0.5, 1)
        is_glass = False
        fname_lower = filename.lower()
        for keyword, mapped_color in MATERIAL_COLORS.items():
            if keyword in fname_lower:
                color = mapped_color
                if "glass" in keyword or "window" in keyword: is_glass = True
                break
        
        mat = create_high_feature_material(f"Mat_{filename}", color, is_glass)
        if obj.data.materials: obj.data.materials[0] = mat
        else: obj.data.materials.append(mat)

def look_at(obj, target_pos):
    """Rotates camera to look at target vector."""
    direction = target_pos - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clamp_to_room(position, margin=0.3):
    """Clamps camera position to stay within room boundaries with margin."""
    x = max(ROOM_MIN.x + margin, min(position[0], ROOM_MAX.x - margin))
    y = max(ROOM_MIN.y + margin, min(position[1], ROOM_MAX.y - margin))
    z = max(ROOM_MIN.z + margin, min(position[2], ROOM_MAX.z - margin))
    return mathutils.Vector((x, y, z))

def generate_focus_orbit(target_obj, frames_list, images_dir, start_index, num_frames=30):
    """
    Generates a tight spiral orbit around a specific object to capture details.
    Ensures camera stays within room boundaries.
    """
    # Get object location and dimensions
    center = target_obj.location
    
    # Calculate maximum safe radius based on room boundaries and object position
    max_radius_x = min(center.x - ROOM_MIN.x, ROOM_MAX.x - center.x) - 0.5
    max_radius_y = min(center.y - ROOM_MIN.y, ROOM_MAX.y - center.y) - 0.5
    safe_radius = min(max_radius_x, max_radius_y, 1.2)  # Cap at 1.2m max
    safe_radius = max(0.6, safe_radius)  # Minimum 0.6m to avoid being too close
    
    print(f"Generating focus scan for: {target_obj.name} at {center}, radius: {safe_radius:.2f}m")

    for i in range(num_frames):
        t = i / num_frames
        angle = t * 2 * math.pi
        
        # Spiral height: Go from High -> Low to see top and under-sides
        # Start at 2.0m (looking down), end at 0.8m (looking level)
        # Constrain z to be within room height
        desired_z = 2.0 - (t * 1.2)
        current_z = max(ROOM_MIN.z + 0.5, min(desired_z, ROOM_MAX.z - 0.3))
        
        # Calculate Camera Position
        cam_x = center.x + math.cos(angle) * safe_radius
        cam_y = center.y + math.sin(angle) * safe_radius
        
        # Clamp to room boundaries with margin
        cam = bpy.context.scene.camera
        cam.location = clamp_to_room((cam_x, cam_y, current_z))
        
        # Look specifically at the object center (not the room center)
        # We add a slight Z-offset to look at the 'mass' of the object, not its feet
        look_target = center + mathutils.Vector((0, 0, 0.5))
        look_at(cam, look_target)
        
        # Render
        render_frame(start_index + i, cam, images_dir, frames_list)

def main():
    reset_scene()
    setup_render_engine()
    import_models()
    setup_lighting()
    
    images_dir = os.path.join(OUTPUT_DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 20 # 20mm is good for room scale
    
    frames = []
    
    # === STRATEGY 1: PERIMETER WALK (100 Frames) ===
    # Walk around the walls, looking at the center
    for i in range(100):
        t = i / 100.0
        angle = t * 2 * math.pi
        
        # Oval path slightly smaller than room limits
        radius_x = (ROOM_MAX.x - ROOM_MIN.x) * 0.35  # Reduced from 0.4 for safety
        radius_y = (ROOM_MAX.y - ROOM_MIN.y) * 0.35
        
        x = CENTER.x + math.cos(angle) * radius_x
        y = CENTER.y + math.sin(angle) * radius_y
        z = 1.6 # Eye level
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look at a point slightly offset from center to create parallax
        look_target = CENTER + mathutils.Vector((math.cos(angle*2)*0.5, math.sin(angle*2)*0.5, -0.5))
        look_at(cam, look_target)
        
        render_frame(i, cam, images_dir, frames)

    # === STRATEGY 2: LOW "DETAIL" PASS (100 Frames) ===
    # Move randomly inside the room at low height, looking slightly UP or STRAIGHT
    for i in range(100):
        # Random position within inner bounds (with safety margin)
        x = random.uniform(ROOM_MIN.x + 0.8, ROOM_MAX.x - 0.8)
        y = random.uniform(ROOM_MIN.y + 0.8, ROOM_MAX.y - 0.8)
        z = random.uniform(0.5, 1.0) # Knee to Waist height
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look at a random point nearby, but generally level or up
        target_x = x + random.uniform(-1, 1)
        target_y = y + random.uniform(-1, 1)
        target_z = random.uniform(0.8, 1.5) # Look slightly up
        
        look_at(cam, mathutils.Vector((target_x, target_y, target_z)))
        render_frame(100 + i, cam, images_dir, frames)

    # === STRATEGY 3: TOP-DOWN FILLER (100 Frames) ===
    # High up, looking down. Fills floor holes.
    for i in range(100):
        # Zig Zag pattern
        t = i / 100.0
        x = ROOM_MIN.x + 0.8 + (ROOM_MAX.x - ROOM_MIN.x - 1.6) * t
        y = CENTER.y + math.sin(t * 10 * math.pi) * 1.2  # Reduced amplitude
        z = 2.5 # Near Ceiling
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look specifically at the floor ahead
        look_at(cam, mathutils.Vector((x, y, 0.0)))
        render_frame(200 + i, cam, images_dir, frames)

    # === STRATEGY 4: OBJECT FOCUS ORBITS ===
    # Automatically find furniture and orbit it
    
    # 1. Identify objects of interest based on names
    keywords = ["chair", "table", "sofa", "tv", "desk"]
    target_objects = []
    
    for obj in bpy.context.scene.objects:
        # Check if object name contains any keyword (case insensitive)
        if any(k in obj.name.lower() for k in keywords):
            target_objects.append(obj)
            
    # 2. Generate orbits for each found object
    frame_counter = 300 # Assuming you already rendered 0-299 in previous steps
    frames_per_object = 25
    
    for obj in target_objects:
        generate_focus_orbit(
            target_obj=obj, 
            frames_list=frames, 
            images_dir=images_dir, 
            start_index=frame_counter, 
            num_frames=frames_per_object
        )
        frame_counter += frames_per_object

    # Save JSON in Blender/NeRF format split into train/test
    # 2DGS expects transforms_train.json and transforms_test.json
    json_data = {"camera_angle_x": cam.data.angle_x, "frames": frames}
    
    # Split: 90% train, 10% test
    num_frames = len(frames)
    num_test = max(1, num_frames // 10)
    test_indices = set(range(0, num_frames, num_frames // num_test))[:num_test]
    
    train_frames = [f for i, f in enumerate(frames) if i not in test_indices]
    test_frames = [f for i, f in enumerate(frames) if i in test_indices]
    
    train_data = {"camera_angle_x": cam.data.angle_x, "frames": train_frames}
    test_data = {"camera_angle_x": cam.data.angle_x, "frames": test_frames}
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_test.json"), 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Done! Generated {len(train_frames)} train and {len(test_frames)} test frames")


def render_frame(index, cam, images_dir, frames_list):
    bpy.context.view_layer.update()
    filename = f"frame_{index:04d}.png"
    filepath = os.path.join(images_dir, filename)
    bpy.context.scene.render.filepath = filepath
    
    # Suppress output for speed
    with open(os.devnull, 'w') as fnull:
         bpy.ops.render.render(write_still=True)
            
    # Standard NeRF/3DGS matrix format
    matrix = cam.matrix_world
    frames_list.append({
        "file_path": f"images/frame_{index:04d}",  # No .png extension
        "transform_matrix": [list(row) for row in matrix]
    })
    print(f"Rendered {filename}")

if __name__ == "__main__":
    main()