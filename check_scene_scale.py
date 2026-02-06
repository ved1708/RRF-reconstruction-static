import os
import numpy as np
from sionna.rt import load_scene

def check_scene_bbox(xml_path):
    print(f"--- Checking Scene Scale: {xml_path} ---")
    
    # 1. Load Scene
    try:
        scene = load_scene(xml_path)
    except Exception as e:
        print(f"Error loading scene: {e}")
        return

    # 2. Access the Raw Mitsuba Scene (Physics Backend)
    # The 'mi_scene' object holds the raw geometry data
    try:
        bbox = scene.mi_scene.bbox()
        
        # Extract Min/Max coordinates
        # These are usually Dr.Jit or Mitsuba vector types, so we convert to string/list
        b_min = np.array(bbox.min)
        b_max = np.array(bbox.max)
        
        dims = b_max - b_min
        center = (b_max + b_min) / 2
        
    except AttributeError:
        print("Could not access 'mi_scene'. Your Sionna version might be very old or very new.")
        return

    # 3. Print Results
    print("\n" + "="*60)
    print(f"{'METRIC':<20} | {'VALUE (Meters)':<30}")
    print("-" * 60)
    print(f"{'Min Coordinates':<20} | {b_min}")
    print(f"{'Max Coordinates':<20} | {b_max}")
    print("-" * 60)
    print(f"{'Total Width (X)':<20} | {dims[0]:.4f} m")
    print(f"{'Total Depth (Y)':<20} | {dims[1]:.4f} m")
    print(f"{'Total Height (Z)':<20} | {dims[2]:.4f} m")
    print("="*60)

    # 4. Diagnosis
    print("\n--- DIAGNOSIS ---")
    
    # Check X/Y dimensions (Room footprint)
    max_dim = max(dims[0], dims[1])
    
    if max_dim < 0.1:
        print("❌ CRITICAL: Scene is TINY (millimeters).") 
        print("   Action: In Blender, set Unit Scale to 1.0 and Re-export.")
    elif max_dim > 500.0:
        print("⚠️ WARNING: Scene is HUGE (> 500m).")
        print("   Action: Confirm this is correct. If this is a single room, your units are likely Centimeters.")
    elif abs(b_min[2]) > 0.1 and abs(b_min[2] - 0.0) > 0.1:
        # If the floor is not at Z=0
        print(f"ℹ️ NOTE: Your floor is at Z = {b_min[2]:.2f}m, not 0.0m.")
        print("   Action: This is fine, but adjust your Transmitter Z-height accordingly.")
    else:
        print("✅ SUCCESS: Dimensions look realistic for an indoor room.")

if __name__ == "__main__":
    # Ensure this matches your actual XML filename
    check_scene_bbox("/tf/Project_1/room_5x3x3.xml")