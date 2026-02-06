
import sionna
import sionna.rt as rt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

xml_path = "/tf/Project_1/room_5x3x3.xml"

print(f"Loading scene: {xml_path}...")
scene = rt.load_scene(xml_path)

print("Initial radio materials:", scene.radio_materials.keys())

if "mat_grey" in scene.radio_materials:
    print(f"Initial mat_grey object: {scene.radio_materials['mat_grey']}")
    if "floor" in scene.objects:
        print(f"Floor material object: {scene.objects['floor'].radio_material}")
        print(f"Are they the same? {scene.radio_materials['mat_grey'] is scene.objects['floor'].radio_material}")

# Try adding the material
mat_grey = rt.RadioMaterial("mat_grey", relative_permittivity=5.31, conductivity=0.04)
scene.radio_materials["mat_grey"] = mat_grey

print("Radio materials after addition:", scene.radio_materials.keys())

if "floor" in scene.objects:
    print(f"Floor material object after update: {scene.objects['floor'].radio_material}")
    print(f"Is it the new mat_grey? {scene.objects['floor'].radio_material is mat_grey}")

# Check object materials
if "floor" in scene.objects:
    obj = scene.objects["floor"]
    print(f"Floor object: {obj}")
    # Check if we can access its material property
    try:
        print(f"Floor radio material: {obj.radio_material}")
    except Exception as e:
        print(f"Error accessing floor.radio_material: {e}")

# Try to trigger the check
try:
    # scene.scene_geometry_updated() # This might be needed or not
    # The error happens during compute_paths which calls _check_scene
    # We can call _check_scene directly if it's accessible, or just try compute_paths
    print("Attempting to compute paths (minimal)...")
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", vertical_spacing=0.5, horizontal_spacing=0.5)
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", vertical_spacing=0.5, horizontal_spacing=0.5)
    scene.add(rt.Transmitter(name="tx", position=[1,1,1]))
    scene.add(rt.Receiver(name="rx", position=[2,2,2]))
    scene.compute_paths(max_depth=1)
    print("Compute paths successful")
except Exception as e:
    print(f"Error during compute_paths: {e}")
