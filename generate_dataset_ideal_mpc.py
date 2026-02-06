import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import cv2  # Removed dependency
from scipy.spatial.transform import Rotation 
from scipy.ndimage import map_coordinates # Added for interpolation
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera as SionnaCamera, RadioMaterial
import sionna

# --- GPU Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# --- COLMAP Structures ---
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

# --- Geometric Helper Functions ---
def euler_to_quaternion(euler):
    # rotation from colmap camera default(initial direction) to sionna array initial direction
    R_posz2posx = Rotation.from_euler('ZYX', [-np.pi/2,0.0,-np.pi/2])
    # rotation of sionna from array initial direction to sampling direction
    yaw, pitch, roll = euler 
    R_posx2array = Rotation.from_euler('ZYX',[yaw, pitch, roll]) 
    # For intrinsic rotations, the rightmost rotation matrix corresponds to the first rotation applied.
    R_w2c =  R_posx2array * R_posz2posx
    R_c2w = R_w2c.inv()
    q = R_c2w.as_quat()
    # colmap requires qw qx qy qz scalar first quaternion
    qvec_c2w = [q[3],q[0],q[1],q[2]] 
    return R_c2w, qvec_c2w

def calculate_camera_intrinsics(width, height, focal_length):
    fx = focal_length
    fy = focal_length
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

def save_intrinsics_text(path, cameras):
    with open(path, "w") as fid:
        for cam_id, cam in cameras.items():
            params_str = " ".join(map(str, cam.params))
            fid.write(f"{cam_id} {cam.model} {cam.width} {cam.height} {params_str}\n")

def save_extrinsics_text(path, images):
    with open(path, "w") as fid:
        for img_id, img in images.items():
            qvec_str = " ".join(map(str, img.qvec))
            tvec_str = " ".join(map(str, img.tvec))
            fid.write(f"{img_id} {qvec_str} {tvec_str} {img.camera_id} {img.name}\n")
            fid.write("\n")

# --- Image Transform Helpers ---
def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]
    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)
    return out

class Equirectangular:
    def __init__(self, img):
        self._img = img
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        
        # Scipy Rotation replacement for cv2.Rodrigues
        r1 = Rotation.from_rotvec(y_axis * np.radians(THETA))
        R1 = r1.as_matrix()
        
        r2 = Rotation.from_rotvec(np.dot(R1, x_axis) * np.radians(PHI))
        R2 = r2.as_matrix()
        
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        
        # Scipy map_coordinates replacement for cv2.remap
        # XY is (H, W, 2) where XY[..., 0] is X (col), XY[..., 1] is Y (row)
        map_cols = XY[..., 0] # indices for width
        map_rows = XY[..., 1] # indices for height
        
        # Stack coordinates for map_coordinates: (2, H, W) -> [rows, cols]
        coords = np.stack([map_rows, map_cols], axis=0)
        
        out_channels = []
        for i in range(self._img.shape[2]):
            # map_coordinates handles interpolation. mode='wrap' behaves like BORDER_WRAP?
            # cv2.BORDER_WRAP means cyclic. scipy 'wrap' means cyclic.
            out_channels.append(map_coordinates(self._img[..., i], coords, order=1, mode='wrap'))
            
        persp = np.stack(out_channels, axis=-1)
        return persp

def equirectangular_to_perspective(img, fov, theta, phi, height, width):
    eq = Equirectangular(img)
    return eq.GetPerspective(FOV=fov, THETA=-theta, PHI=90-phi, height=height, width=width)

# --- Core MPC Spectrum Generation ---
def plot_spatial_spectrum(path_instance, image_scale=3, kernel_size=3, kernel_sigma=3):
    theta_r = path_instance.theta_r.numpy()  
    phi_r = path_instance.phi_r.numpy()      
    intensities = path_instance.a.numpy()  
    
    # Check if empty (no paths)
    if intensities.size == 0:
        return np.ones((180*image_scale, 360*image_scale)) * -160.0 # Return noise floor

    img = np.zeros((360*image_scale, 180*image_scale))
 
    sigma_x, sigma_y = kernel_sigma, kernel_sigma
    theta = theta_r[0, 0, 0, :]*180/np.pi
    phi = phi_r[0, 0, 0, :]*180/np.pi
    amps = np.abs(intensities[0, 0, 0, 0, 0, :,0])
 
    size_x = int(kernel_size * sigma_x) | 1  
    size_y = int(kernel_size * sigma_y) | 1  
    x = np.linspace(-size_x // 2, size_x // 2, size_x)
    y = np.linspace(-size_y // 2, size_y // 2, size_y)
    x, y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    for idx, intensity in enumerate(amps):
        if intensity > 1e-9: # Threshold for meaningful paths
            path_dot = gauss_kernel*intensity/np.sum(gauss_kernel) 
            phi_idx = int(-phi[idx] + 180)*image_scale
            theta_idx = int(theta[idx])*image_scale
            xmin = max(0, phi_idx - size_x // 2)
            xmax = min(360*image_scale, phi_idx + size_x // 2 + 1)
            ymin = max(0, theta_idx - size_y // 2)
            ymax = min(180*image_scale, theta_idx + size_y // 2 + 1)
 
            gauss_xmin = max(0, size_x // 2 - phi_idx)
            gauss_xmax = min(size_x, 360*image_scale - phi_idx + size_x // 2)
            gauss_ymin = max(0, size_y // 2 - theta_idx)
            gauss_ymax = min(size_y, 180*image_scale - theta_idx + size_y // 2)
 
            # Ensure indices match
            if (xmax > xmin) and (ymax > ymin):
                target_slice = img[xmin:xmax, ymin:ymax]
                source_slice = path_dot[gauss_xmin:gauss_xmax, gauss_ymin:gauss_ymax]
                
                # Double check shapes match before adding
                if target_slice.shape == source_slice.shape:
                    img[xmin:xmax, ymin:ymax] += source_slice

    non_zero_mask = img != 0.0
    zero_mask = img == 0.0

    img[non_zero_mask] = 10*np.log10(img[non_zero_mask])  
    img[zero_mask] = np.min(img[non_zero_mask])-10 if np.any(non_zero_mask) else -160
    
    return img

def generate_ideal_dataset(scene, rx_locs, tx_loc, output_dir):
    spectrum_dir = os.path.join(output_dir, 'spectrum')
    cameras_file = os.path.join(output_dir, 'cameras.txt')
    images_file = os.path.join(output_dir, 'images.txt')
    
    os.makedirs(spectrum_dir, exist_ok=True)
    
    if 'tx' in scene.objects: scene.remove('tx')
    scene.add(Transmitter(name='tx', position=tx_loc))

    width, height = 300, 200 
    fov = 90
    
    # Check if we should use tfmath or numpy
    # np.tan is safe.
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2)) 
    fx, fy, cx, cy = calculate_camera_intrinsics(width, height, focal_length)
    camera_id = 1
    cameras = {camera_id: Camera(camera_id, "PINHOLE", width, height, [fx, fy, cx, cy])}
    images = {}
    
    jet_colormap = plt.get_cmap('jet')

    # Add Receiver ONCE
    if 'rx' in scene.objects: scene.remove('rx')
    rx = Receiver(name='rx', position=[0,0,0])
    scene.add(rx)

    # 1. Determine Global Min/Max for normalization
    print("Computing global stats (Min/Max dB)...")
    spec_max = -np.inf
    spec_min = np.inf
    
    # Sample a subset to save time
    sample_indices = np.linspace(0, len(rx_locs)-1, min(20, len(rx_locs)), dtype=int)
    
    for idx in sample_indices:
        rx.position = rx_locs[idx]
        
        # Scat_keep_prob=0.5 for detailed MPC
        # Using 1 depth to capture basic paths first
        paths = scene.compute_paths(max_depth=1, reflection=True, diffraction=False, scattering=True, 
                                    scat_keep_prob=0.5, num_samples=int(1e5))
        
        # Check if paths found
        if paths.a.shape[-2] == 0:
            continue

        spec_dB = plot_spatial_spectrum(paths).T
        if np.max(spec_dB) > spec_max: spec_max = np.max(spec_dB)
        if np.min(spec_dB) < spec_min: spec_min = np.min(spec_dB)
    
    # Fallback if no paths found in sample
    if spec_max == -np.inf:
        spec_max = -40
        spec_min = -160
        
    print(f"Global Max dB: {spec_max}, Global Min dB: {spec_min}")

    # 2. Generate Dataset
    image_counter = 1
    for i, rx_loc in enumerate(rx_locs):
        print(f"Generating view {i+1}/{len(rx_locs)} at {rx_loc}")
        
        rx.position = rx_loc
        
        # Higher sample count for final images
        # Reduced depth and samples to avoid OptiX compilation error/timeout
        paths = scene.compute_paths(max_depth=1, reflection=True, diffraction=False, scattering=True, 
                                    scat_keep_prob=0.5, num_samples=int(5e5))
        
        spec_dB = plot_spatial_spectrum(paths).T
        
        # Normalize
        spec_norm = np.clip((spec_dB - spec_min) / (spec_max - spec_min + 1e-9), 0, 1)
        
        # Apply Jet Colormap (returns RGBA, take RGB)
        eq_img_colored = jet_colormap(spec_norm)[:, :, :3] 

        # Generate 4 orientations
        for angle in [-np.pi/2, 0, np.pi/2, np.pi]:
            # Convert Equirectangular to Perspective
            persp_img = equirectangular_to_perspective(eq_img_colored, fov, angle*180/np.pi, 90, height, width)
            # Ensure float32 0-1
            persp_img = np.clip(persp_img, 0, 1)
            
            # Save Image
            img_filename = f"{image_counter:05d}.png"
            img_path = os.path.join(spectrum_dir, img_filename)
            plt.imsave(img_path, persp_img)
            
            # Save Pose
            orientation = [angle, 0, 0] # Yaw, Pitch, Roll
            R_c2w, qvec_c2w = euler_to_quaternion(orientation)
            tvec_w2c = rx_loc
            tvec_c2w = -R_c2w.apply(tvec_w2c)
            
            images[image_counter] = colmap_Image(image_counter, qvec_c2w, tvec_c2w, camera_id, img_filename, [], [])
            image_counter += 1
            
    print("")

    save_intrinsics_text(cameras_file, cameras)
    save_extrinsics_text(images_file, images)
    print("Dataset generation complete.")

if __name__ == "__main__":
    # --- Scene Setup ---
    # We must assume the fixed XML is present
    if not os.path.exists("room_5x3x3_fixed.xml"):
        print("Error: room_5x3x3_fixed.xml not found. Please ensure it exists.")
        exit(1)
        
    scene = load_scene("room_5x3x3_fixed.xml")
    
    # Use 28GHz to match previous script unless strict tutorial adherence requires 60GHz.
    scene.frequency = 60e9 
    scene.synthetic_array = True 
    wavelength = 299792458 / scene.frequency
    
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                                vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                                vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)

    # Define Materials with Scattering (Required for Ideal MPC)
    global_scattering_coeff = 4
    
    # Create materials but check if they exist (Sionna might load them from XML with generic names)
    # The snippet creates new RadioMaterials with scattering properties.
    
    mat_concrete = RadioMaterial("mat_concrete_scat", relative_permittivity=5.24, conductivity=0.0462, 
                                 scattering_coefficient=0.1*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=5))
    mat_wood = RadioMaterial("mat_wood_scat", relative_permittivity=1.99, conductivity=0.0047, 
                                 scattering_coefficient=0.2*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=3))
    mat_glass = RadioMaterial("mat_glass_scat", relative_permittivity=6.31, conductivity=0.0036, 
                                 scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))
    mat_metal = RadioMaterial("mat_metal_scat", relative_permittivity=1, conductivity=1e7, 
                                 scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))

    # Add materials safely
    for mat in [mat_concrete, mat_wood, mat_glass, mat_metal]:
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    # Re-assign materials to objects
    # This overwrites the materials assigned by load_scene from fixed XML
    print("Re-assigning materials for Scattering...")
    for obj_name, obj in scene.objects.items():
        # Clean naming matching
        name = obj_name.lower()
        if "floor" in name or "walls" in name or "ceiling" in name or "pillar" in name:
            obj.radio_material = "mat_concrete_scat"
        elif "furniture" in name or "door" in name:
            obj.radio_material = "mat_wood_scat"
        elif "window" in name:
            obj.radio_material = "mat_glass_scat"
        elif "tv" in name or "led" in name:
            obj.radio_material = "mat_metal_scat"
        else:
            print(f"Warning: Object {obj_name} using default material.")

    # --- Sampling Campaign ---
    # Grid generation for the room (Custom bounds 0.5 to 4.5m x, 0.5 to 2.5m y)
    x_range = np.linspace(0.5, 4.5, 10) 
    y_range = np.linspace(0.5, 2.5, 5)
    z_height = 1.2 # Device height
    rx_locs = [[x, y, z_height] for x in x_range for y in y_range]
    
    tx_pos = [2.5, 1.5, 2.5] # Matched from previous script
    
    print(f"Starting Ideal Beamforming generation for {len(rx_locs)} positions...")
    generate_ideal_dataset(scene, rx_locs, tx_pos, "dataset_custom_scene_ideal_mpc")
