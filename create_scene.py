#!/usr/bin/env python3
"""
Generates all necessary files for a simple room scene for Sionna.

This script creates:
1.  .ply mesh files for a 6m x 3m x 4m room with correct inward-facing normals.
2.  A room.xml scene file for Mitsuba/Sionna that assembles the meshes.
"""
import os
import struct

# --- Helper Functions ---

def write_ply(filename, vertices, faces):
    """
    Writes a binary PLY file with vertices and triangular faces.
    Ensures the output directory exists.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))
        
        # Vertices
        for v in vertices:
            f.write(struct.pack('fff', v[0], v[1], v[2]))
        
        # Faces (triangles)
        for face in faces:
            f.write(struct.pack('B', 3))  # 3 vertices per face
            f.write(struct.pack('iii', face[0], face[1], face[2]))

def create_rectangle(x_range, y_range, z_val):
    """Creates vertices for a rectangle on the XY plane at a given Z height."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    return [
        [x_min, y_min, z_val],
        [x_max, y_min, z_val],
        [x_max, y_max, z_val],
        [x_min, y_max, z_val],
    ]

def create_box(x_range, y_range, z_range):
    """Creates vertices and faces for a box with inward-facing normals."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
    ]
    
    faces = [
        [0, 3, 2], [0, 2, 1], # Bottom (-z)
        [4, 5, 6], [4, 6, 7], # Top (+z)
        [0, 1, 5], [0, 5, 4], # Front (-y)
        [2, 3, 7], [2, 7, 6], # Back (+y)
        [0, 4, 7], [0, 7, 3], # Left (-x)
        [1, 2, 6], [1, 6, 5], # Right (+x)
    ]
    return vertices, faces

# --- Main Scene Creation ---

def create_scene_geometry(output_dir):
    """
    Generates all .ply files for the room.
    Normals are oriented to point inwards into the room.
    """
    print(f"Generating mesh files in: {output_dir}")

    # Floor (6m x 3m at z=0). Normal points up (+z).
    floor_v = create_rectangle((0, 6), (0, 3), 0)
    floor_f = [[0, 1, 2], [0, 2, 3]]
    write_ply(os.path.join(output_dir, 'floor.ply'), floor_v, floor_f)
    print("✓ floor.ply")

    # Ceiling (6m x 3m at z=4). Normal points down (-z).
    ceil_v = create_rectangle((0, 6), (0, 3), 4)
    ceil_f = [[0, 2, 1], [0, 3, 2]] # Flipped for inward normal
    write_ply(os.path.join(output_dir, 'ceiling.ply'), ceil_v, ceil_f)
    print("✓ ceiling.ply")

    # Wall 1 (y=0, 6m x 4m). Normal points in (+y).
    wall1_v = [[0, 0, 0], [6, 0, 0], [6, 0, 4], [0, 0, 4]]
    wall1_f = [[0, 2, 1], [0, 3, 2]] # Flipped for inward normal
    write_ply(os.path.join(output_dir, 'wall1.ply'), wall1_v, wall1_f)
    print("✓ wall1.ply")

    # Wall 2 (y=3, 6m x 4m). Normal points in (-y).
    wall2_v = [[0, 3, 0], [6, 3, 0], [6, 3, 4], [0, 3, 4]]
    wall2_f = [[0, 1, 2], [0, 2, 3]]
    write_ply(os.path.join(output_dir, 'wall2.ply'), wall2_v, wall2_f)
    print("✓ wall2.ply")

    # Wall 3 (x=0, 3m x 4m). Normal points in (+x).
    wall3_v = [[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]]
    wall3_f = [[0, 1, 2], [0, 2, 3]]
    write_ply(os.path.join(output_dir, 'wall3.ply'), wall3_v, wall3_f)
    print("✓ wall3.ply")

    # Wall 4 (x=6, 3m x 4m). Normal points in (-x).
    wall4_v = [[6, 0, 0], [6, 3, 0], [6, 3, 4], [6, 0, 4]]
    wall4_f = [[0, 2, 1], [0, 3, 2]] # Flipped for inward normal
    write_ply(os.path.join(output_dir, 'wall4.ply'), wall4_v, wall4_f)
    print("✓ wall4.ply")

    # --- Add Table ---
    # Table Top (1.2m x 0.8m, 5cm thick, at height 0.75m)
    # Centered at (4.0, 1.5)
    table_v, table_f = create_box((3.4, 4.6), (1.1, 1.9), (0.7, 0.75))
    write_ply(os.path.join(output_dir, 'table_top.ply'), table_v, table_f)
    print("✓ table_top.ply")

    # Table Legs (5cm x 5cm, 70cm high)
    leg_v, leg_f = create_box((3.4, 3.45), (1.1, 1.15), (0, 0.7))
    write_ply(os.path.join(output_dir, 'table_leg1.ply'), leg_v, leg_f)
    leg_v, leg_f = create_box((4.55, 4.6), (1.1, 1.15), (0, 0.7))
    write_ply(os.path.join(output_dir, 'table_leg2.ply'), leg_v, leg_f)
    leg_v, leg_f = create_box((3.4, 3.45), (1.85, 1.9), (0, 0.7))
    write_ply(os.path.join(output_dir, 'table_leg3.ply'), leg_v, leg_f)
    leg_v, leg_f = create_box((4.55, 4.6), (1.85, 1.9), (0, 0.7))
    write_ply(os.path.join(output_dir, 'table_leg4.ply'), leg_v, leg_f)
    print("✓ 4 table legs")
    
    print("\n✅ All mesh files created successfully.")

def create_scene_xml(output_path):
    """
    Generates the room.xml file that references the .ply meshes.
    """
    print(f"Generating scene definition file: {output_path}")

    xml_content = f"""<scene version="2.1.0">
    <!-- Emitter for basic lighting -->
    <emitter type="constant">
        <rgb name="radiance" value="1.0"/>
    </emitter>

    <!-- Materials -->
    <bsdf type="diffuse" id="mat_white">
        <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
    </bsdf>
    <bsdf type="diffuse" id="mat_grey">
        <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
    </bsdf>
    <bsdf type="diffuse" id="mat_wood">
        <rgb name="reflectance" value="0.4, 0.2, 0.1"/>
    </bsdf>

    <!-- Floor -->
    <shape type="ply" id="floor">
        <string name="filename" value="floor.ply"/>
        <ref id="mat_grey" name="bsdf"/>
    </shape>

    <!-- Ceiling -->
    <shape type="ply" id="ceiling">
        <string name="filename" value="ceiling.ply"/>
        <ref id="mat_white" name="bsdf"/>
    </shape>

    <!-- Walls -->
    <shape type="ply" id="wall1">
        <string name="filename" value="wall1.ply"/>
        <ref id="mat_white" name="bsdf"/>
    </shape>
    <shape type="ply" id="wall2">
        <string name="filename" value="wall2.ply"/>
        <ref id="mat_white" name="bsdf"/>
    </shape>
    <shape type="ply" id="wall3">
        <string name="filename" value="wall3.ply"/>
        <ref id="mat_white" name="bsdf"/>
    </shape>
    <shape type="ply" id="wall4">
        <string name="filename" value="wall4.ply"/>
        <ref id="mat_white" name="bsdf"/>
    </shape>

    <!-- Table -->
    <shape type="ply" id="table_top">
        <string name="filename" value="table_top.ply"/>
        <ref id="mat_wood" name="bsdf"/>
    </shape>
    <shape type="ply" id="table_leg1">
        <string name="filename" value="table_leg1.ply"/>
        <ref id="mat_wood" name="bsdf"/>
    </shape>
    <shape type="ply" id="table_leg2">
        <string name="filename" value="table_leg2.ply"/>
        <ref id="mat_wood" name="bsdf"/>
    </shape>
    <shape type="ply" id="table_leg3">
        <string name="filename" value="table_leg3.ply"/>
        <ref id="mat_wood" name="bsdf"/>
    </shape>
    <shape type="ply" id="table_leg4">
        <string name="filename" value="table_leg4.ply"/>
        <ref id="mat_wood" name="bsdf"/>
    </shape>
</scene>
"""
    with open(output_path, 'w') as f:
        f.write(xml_content)
    print("✅ Scene XML file created successfully.")


if __name__ == "__main__":
    # The script will save files relative to its location.
    # We assume it's in Project_1, so files go into Project_1/
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    create_scene_geometry(project_dir)
    create_scene_xml(os.path.join(project_dir, "room.xml"))
    
    print("\n--- Scene Generation Complete ---")
