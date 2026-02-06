import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sionna
from sionna.rt import Transmitter, Receiver, PlanarArray
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
import scipy.spatial.transform as transform

# Configuration
CONFIG = {
    "frequency": 28e9,
    "tx_power": 30,  # dBm
    "tx_pos": [2.5, 1.5, 2.5], # Updated for 5x3x3 room (Center-ish, high)
    "rx_z": 1.2, # Typical device height
    "rx_grid_spacing": 0.2, # Finer grid for small room
    "grid_bounds": {"x_min": 0.5, "x_max": 4.5, "y_min": 0.5, "y_max": 2.5}, # Within 5x3 room bounds
    "num_samples": int(5e4),
    "max_depth": 3,
    "M": 8, # 8x8 array as in tutorial
    "output_dir": "dataset_custom_scene_cbf",
    "n_positions": 50 # Increase positions for dataset
}

# --- Helper Classes ---
class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

class colmap_Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

# --- Helper Functions from Tutorial ---

def euler_to_quaternion(euler):
    # euler is [roll, pitch, yaw] in radians
    # Tutorial uses ZYX convention likely?
    # Actually Sionna orientation is [yaw, pitch, roll] usually (Z, Y, X).
    # But colmap needs w, x, y, z
    r = transform.Rotation.from_euler('xyz', euler) 
    q = r.as_quat() # x, y, z, w
    return r, np.array([q[3], q[0], q[1], q[2]])

def calculate_camera_intrinsics(width, height, focal_length):
    fx = focal_length
    fy = focal_length
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

def compute_angle_matrices(width, height, fov):
    focal = width / (2 * tf.math.tan(np.deg2rad(fov) / 2))
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(height / 2, -height / 2, height)
    xx, yy = np.meshgrid(x, y)
    
    # Compute directions in camera coordinates
    directions = np.stack([(xx / focal), (yy/focal), np.ones_like(xx)], axis=-1)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    
    # Compute angles
    phi = np.arctan2(-directions[..., 0], directions[..., 2]) 
    theta = np.pi/2 - np.arcsin(directions[..., 1]) 
    return theta, phi

def steering_vector(M, theta_grid, phi_grid):
    # Matches tutorial implementation
    theta_grid = tf.cast(theta_grid, tf.float32)
    phi_grid = tf.cast(phi_grid, tf.float32)
    
    v = tf.sin(theta_grid)*tf.sin(phi_grid)
    w = tf.cos(theta_grid)
    
    # Handle both grid (2D) and list (1D) inputs
    if len(theta_grid.shape) == 2:
         v_grid = tf.expand_dims(v,0) # [1, H, W]
         w_grid = tf.expand_dims(w,0)
    else:
         v_grid = tf.expand_dims(v,0) # [1, P]
         w_grid = tf.expand_dims(w,0)

    values = 0.25 + 0.5 * np.arange(M/2)
    y_i = np.concatenate((-values[::-1],values)) 
    z_i = np.concatenate((values[::-1],-values))
    
    y_i_grid, z_i_grid = np.meshgrid(y_i, z_i)
    y_i_grid = np.reshape(y_i_grid.transpose(),(M**2))
    z_i_grid = np.reshape(z_i_grid.transpose(),(M**2))
    
    # Expand to match grid dimensions
    if len(theta_grid.shape) == 2:
        y_i_grid = tf.expand_dims(tf.expand_dims(y_i_grid,-1),-1) # [M*M, 1, 1]
        z_i_grid = tf.expand_dims(tf.expand_dims(z_i_grid,-1),-1)
    else:
        y_i_grid = tf.expand_dims(y_i_grid,-1) # [M*M, 1] - Correct for 1D broadcasting
        z_i_grid = tf.expand_dims(z_i_grid,-1)

    y_i_grid = tf.cast(y_i_grid, tf.float32)
    z_i_grid = tf.cast(z_i_grid, tf.float32)
    
    phase = 2*np.pi*(tf.multiply(y_i_grid,v_grid)+tf.multiply(z_i_grid,w_grid))
    c_phase = tf.complex(tf.zeros_like(phase), phase)
    alpha_i = tf.exp(c_phase)
    return alpha_i

def jet_colormap_convert(array_2d):
    # Normalize to 0-1
    if np.max(array_2d) == np.min(array_2d):
         normalized_array = np.zeros_like(array_2d)
    else:
         normalized_array = (array_2d - np.min(array_2d)) / (np.max(array_2d) - np.min(array_2d))
    
    # Use matplotlib colormap
    import matplotlib
    jet_colormap = matplotlib.colormaps.get_cmap('jet')
    colored_image = jet_colormap(normalized_array)
    colored_image = colored_image[:, :, :3] # Remove alpha
    return colored_image

def save_intrinsics_text(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list\n")
        f.write("# camera_id, model, width, height, params[]\n")
        for cam_id, cam in cameras.items():
            params_str = " ".join(map(str, cam.params))
            f.write(f"{cam_id} {cam.model} {cam.width} {cam.height} {params_str}\n")

def save_extrinsics_text(path, images):
    with open(path, "w") as f:
        f.write("# Image list\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for img_id, img in images.items():
            q = img.qvec
            t = img.tvec
            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img.camera_id} {img.name}\n")
            f.write("\n") 

# --- Main Script ---

def main():
    # Load Custom Scene
    # Note: load_scene validates materials immediately upon loading if they are referenced in the file.
    # If the XML references materials but they are not defined, it might fail inside load_scene before we can add them.
    # However, if the PLY does not have material names, Sionna might complain about default.
    # Our room_5x3x3.xml does NOT define materials, it just includes PLY files.
    # The PLY files have 'material_id' but no material mapping in the XML.
    # Usually Sionna expects <shape> to have <ref id="material_name"/> or similar.
    
    # Workaround: Define a minimal scene with code instead of loading XML if XML is problematic.
    # OR: Load scene with ignore_materials=True if possible? No such option usually.
    # BUT wait, the error is: "Material '' is used by the object 'floor' but not defined."
    # This means 'floor' in XML has an empty material reference? Or defaults to empty?
    
    # Let's try to construct the scene entirely in Python to avoid XML parsing issues with materials.
    
    # Load the fixed XML scene
    print("Loading scene from room_5x3x3_fixed.xml...")
    scene = sionna.rt.load_scene("room_5x3x3_fixed.xml")

    # Define High-Quality Radio Materials
    # We define them here and assign them to the loaded objects to ensure correct RF properties
    itu_concrete = sionna.rt.RadioMaterial("itu_concrete", relative_permittivity=5.31, conductivity=0.0326)
    itu_glass = sionna.rt.RadioMaterial("itu_glass", relative_permittivity=6.27, conductivity=0.0125)
    itu_wood = sionna.rt.RadioMaterial("itu_wood", relative_permittivity=1.99, conductivity=0.0381)
    itu_metal = sionna.rt.RadioMaterial("itu_metal", relative_permittivity=1.0, conductivity=1e7)
    
    # Add materials to scene (optional if we assign directly, but good practice)
    # Check if materials already exist to avoid duplication errors
    for mat in [itu_concrete, itu_glass, itu_wood, itu_metal]:
        if mat.name not in scene.radio_materials:
            scene.add(mat)


    # Map object names to RadioMaterials
    # Note: The XML loaded objects with generic diffuse materials. We override them here.
    obj_mat_mapping = {
        "floor": itu_concrete,
        "ceiling": itu_concrete,
        "walls": itu_concrete,
        "window": itu_glass,
        "door": itu_wood,
        "furniture": itu_wood,
        "furniture_center": itu_wood,
        "pillar": itu_concrete,
        "led_tv": itu_metal
    }

    print("Assigning RadioMaterials to objects...")
    for obj_name, radio_mat in obj_mat_mapping.items():
        try:
            if obj_name in scene.objects:
                # Assign by name to avoid duplicate addition errors if material is already in scene
                scene.get(obj_name).radio_material = radio_mat.name
                
                # Verify assignment
                # print(f"Object {obj_name} material: {scene.get(obj_name).radio_material.name}")
            else:
                print(f"Warning: Object '{obj_name}' found in mapping but not in scene.")
        except Exception as e:
            print(f"Error assigning material to {obj_name}: {e}")

    # Configure Antenna Arrays (Required for compute_paths)
    # Using 'tx_array' and 'rx_array' as per Sionna API
    wavelength = 299792458 / CONFIG["frequency"]
    
    # Create Arrays
    tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                          vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                          vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    
    # Assign to scene
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    scene.frequency = CONFIG["frequency"]
    scene.synthetic_array = True 
    
    # Transmitter
    tx = Transmitter(name="tx", position=CONFIG["tx_pos"])
    scene.add(tx)
    
    # Generate RX positions
    x_min, x_max = CONFIG["grid_bounds"]["x_min"], CONFIG["grid_bounds"]["x_max"]
    y_min, y_max = CONFIG["grid_bounds"]["y_min"], CONFIG["grid_bounds"]["y_max"]
    step = CONFIG["rx_grid_spacing"]
    
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)
    rx_locs = []
    for x in xs:
        for y in ys:
            rx_locs.append([x, y, CONFIG["rx_z"]])
            
    # Output Setup
    data_dir = CONFIG["output_dir"]
    os.makedirs(data_dir, exist_ok=True)
    
    # Matches tutorial folder structure
    spectrum_dir = os.path.join(data_dir, 'spectrum') 
    os.makedirs(spectrum_dir, exist_ok=True)
    
    # Camera Intrinsics
    width, height = 300, 200
    fov = 90
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2))
    fx, fy, cx, cy = calculate_camera_intrinsics(width, height, focal_length)
    
    cameras = {}
    cameras[1] = Camera(1, "PINHOLE", width, height, [fx, fy, cx, cy])
    
    images = {}
    image_counter = 1
    
    n_positions = CONFIG["n_positions"]
    M = CONFIG["M"]
    
    print(f"Generating spatial spectra for first {n_positions} positions using Ideal Beamforming (CBF)...")
    print(f"Frequency: {CONFIG['frequency']/1e9} GHz")
    print(f"Array Size: {M}x{M}")
    
    for i, rx_loc in enumerate(rx_locs):
        if i >= n_positions:
            break
            
        print(f"Processing position {i+1}/{n_positions}: {rx_loc}", end='\r')
        
        # 4 Orientations (NSEW)
        for angle in [-np.pi/2, 0, np.pi/2, np.pi]:
            
            # Orientation setup
            orientation = [angle, 0, 0] 
            rx = Receiver(name='rx', position=rx_loc, orientation=orientation)
            
            # COLMAP Geometry
            # r, q = euler_to_quaternion(orientation)
            # Tutorial: R_c2w, qvec_c2w = euler_to_quaternion(rx.orientation)
            # tvec_w2c = rx_loc
            # tvec_c2w = -R_c2w.apply(tvec_w2c)
            # Sionna RT receiver has orientation property.
            
            # Since we manually set orientation, we can use it.
            r, qvec_colmap = euler_to_quaternion(orientation)
            
            # Calculate tvec for COLMAP (world to camera translation)
            # t_w2c = -R_w2c * C (where C is camera center/rx_loc)
            # But colmap uses R * X + t.
            # R_colmap (w2c) is inverse of R_c2w (orientation).
            R_c2w = r.as_matrix()
            R_w2c = R_c2w.T
            t_w2c = -np.dot(R_w2c, rx_loc)
            
            # However, tutorial code:
            # tvec_w2c = rx_loc 
            # tvec_c2w = -R_c2w.apply(tvec_w2c)
            # This tutorial logic for tvec/qvec seems ... specific. 
            # I will trust standard COLMAP conventions: qvec is R_w2c (scalar first), tvec is t_w2c.
            # Tutorial `euler_to_quaternion` returns qvec corresponding to R_c2w?
            # Usually orientation is C2W.
            # I will use standard conversion: 
            # q_c2w = from_euler
            # R_c2w = q_c2w.matrix
            # R_w2c = R_c2w.T
            # t_w2c = -R_w2c @ center
            # q_w2c = R_w2c.as_quat (w,x,y,z)
            
            # Let's check tutorial again later if tracking fails.
            q_c2w = transform.Rotation.from_euler('xyz', orientation)
            R_c2w_mat = q_c2w.as_matrix()
            R_w2c_mat = R_c2w_mat.T
            t_w2c = -np.dot(R_w2c_mat, rx_loc)
            q_w2c = transform.Rotation.from_matrix(R_w2c_mat).as_quat() # x,y,z,w
            qvec = np.array([q_w2c[3], q_w2c[0], q_w2c[1], q_w2c[2]]) # w,x,y,z
            
            scene.add(rx)
            paths = scene.compute_paths(max_depth=CONFIG["max_depth"], 
                                      num_samples=CONFIG["num_samples"],
                                      scattering=False, diffraction=False, reflection=True) 
            
            # --- CBF/Ideal Beamforming Calculation ---
            theta_grid, phi_grid = compute_angle_matrices(width, height, fov)
            
            spectrum = np.zeros((height, width))
            
            if paths.mask.numpy().any(): # If paths exist
                 # Extract path properties
                 # paths.a: [num_tx, num_tx_ant, num_rx, num_rx_ant, num_paths, pol, pol]
                 # We assume SISO, so indices 0.
                 alpha = paths.a[0,0,0,0,0,:,0] # Complex path coefficients [P]
                 theta_r = paths.theta_r[0,0,0,:] # [P]
                 phi_r = paths.phi_r[0,0,0,:]     # [P]
                 
                 # Valid paths
                 valid = tf.abs(alpha) > 1e-12
                 alpha_valid = tf.boolean_mask(alpha, valid)
                 theta_valid = tf.boolean_mask(theta_r, valid)
                 phi_valid = tf.boolean_mask(phi_r, valid)
                 
                 if tf.size(alpha_valid) > 0:
                     # 1. Compute synthetic array response vector y [M*M]
                     # sv_paths: [M*M, num_valid_paths]
                     sv_paths = steering_vector(M, theta_valid, phi_valid)
                     
                     # y = sum(alpha * sv_paths) over paths
                     # sv_paths is [M*M, P], alpha is [P]
                     # y[k] = sum_p alpha[p] * sv[k, p]
                     y = tf.reduce_sum(sv_paths * alpha_valid, axis=1) # [M*M]
                     
                     # 2. Compute beamforming weights for all pixels w [M*M, H, W]
                     w_grid = steering_vector(M, theta_grid, phi_grid)
                     
                     # 3. Compute spectrum P = |w^H * y|^2
                     # y: [M*M], w: [M*M, H, W]
                     # w^H * y = sum(conj(w) * y, axis=0)
                     beamformed = tf.reduce_sum(tf.math.conj(w_grid) * tf.expand_dims(tf.expand_dims(y, -1), -1), axis=0)
                     P_spec = tf.abs(beamformed)**2
                     spectrum = 10 * np.log10(P_spec.numpy() + 1e-30)
            
            # Normalize and Colorize
            img_rgb_uint8 = (jet_colormap_convert(spectrum) * 255).astype(np.uint8)
            
            # Save Image
            img_filename = f"{image_counter:05d}.png"
            Image.fromarray(img_rgb_uint8).save(os.path.join(spectrum_dir, img_filename))
            
            # Store COLMAP data
            images[image_counter] = colmap_Image(image_counter, qvec, t_w2c, 1, img_filename, [], [])
            
            image_counter += 1
            scene.remove('rx')
    
    scene.remove('tx')
    
    # Save COLMAP text files
    save_intrinsics_text(os.path.join(data_dir, 'cameras.txt'), cameras)
    save_extrinsics_text(os.path.join(data_dir, 'images.txt'), images)
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    main()
