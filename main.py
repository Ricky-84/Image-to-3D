import torch
import numpy as np
import trimesh
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import os
from rembg import remove
import io

model_id = "Intel/dpt-hybrid-midas"
feature_extractor = DPTFeatureExtractor.from_pretrained(model_id)
depth_model = DPTForDepthEstimation.from_pretrained(model_id)

def preprocess_image(image_path):
    # Use rembg to remove background with AI
    with open(image_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)

    # Convert to image with alpha channel (RGBA)
    img_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    return img_no_bg

def estimate_depth(image):
    # Resize the image to its original dimensions to avoid feature_extractor's default resizing
    original_size = image.size  # (width, height)
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize(original_size, Image.Resampling.LANCZOS)
    inputs = feature_extractor(images=image_resized, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    # Resize depth map back to original dimensions if needed
    depth = np.array(Image.fromarray(depth).resize(original_size, Image.Resampling.LANCZOS))
    return depth
def generate_3d_mesh(depth_map, image, scale=0.2, threshold=0.5):
    h, w = depth_map.shape
    img_array = np.array(image.convert("RGBA"))  # Ensure RGBA for mask
    mask = img_array[:, :, 3] / 255.0

    # Debug: Print dimensions to verify
    print(f"Depth map shape: {depth_map.shape}, Mask shape: {mask.shape}")

    vertices = []
    vertex_map = {}
    for y in range(h):
        for x in range(w):
            if mask[y, x] > threshold:
                z = depth_map[y, x] * scale
                vertex_idx = len(vertices)
                # Center the coordinates
                centered_x = x - w / 2
                centered_y = y - h / 2
                vertices.append([centered_x, centered_y, z])
                vertex_map[(x, y)] = vertex_idx

    if not vertices:
        print("No vertices generated. Creating fallback mesh.")
        return trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])

    max_x = max(v[0] for v in vertices)
    min_x = min(v[0] for v in vertices)
    mid_x = (max_x + min_x) / 2

    mirrored_vertex_map = {}
    vertices_mirrored = []
    for x, y, z in vertices:
        # Use the original centered x, y, z for mirroring
        original_x = x + w / 2  # Reverse the centering to get original x
        mirrored_x = 2 * mid_x - original_x
        vertex_idx = len(vertices) + len(vertices_mirrored)
        vertices_mirrored.append([mirrored_x - w / 2, y, z])  # Re-center the mirrored x
        mirrored_vertex_map[(mirrored_x, y + h / 2)] = vertex_idx  # Adjust key to match centered y
    vertices.extend(vertices_mirrored)

    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            if all((pt in vertex_map) for pt in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]):
                v0 = vertex_map[(x, y)]
                v1 = vertex_map[(x + 1, y)]
                v2 = vertex_map[(x, y + 1)]
                v3 = vertex_map[(x + 1, y + 1)]
                faces.extend([[v0, v1, v2], [v1, v3, v2]])

    for y in range(h - 1):
        for x in range(w - 1):
            mx = 2 * mid_x - (x - w / 2)
            mx_next = 2 * mid_x - ((x + 1) - w / 2)
            my = y - h / 2
            if all((pt in mirrored_vertex_map) for pt in [(mx, my), (mx_next, my), (mx, my + 1), (mx_next, my + 1)]):
                v0 = mirrored_vertex_map[(mx, my)]
                v1 = mirrored_vertex_map[(mx_next, my)]
                v2 = mirrored_vertex_map[(mx, my + 1)]
                v3 = mirrored_vertex_map[(mx_next, my + 1)]
                faces.extend([[v0, v2, v1], [v1, v2, v3]])

    if not faces:
        print("No faces generated. Creating fallback mesh.")
        return trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def extrude_mesh(mesh, extrusion_depth=50.0):
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Invalid mesh passed for extrusion.")

    original_vertices = mesh.vertices
    original_faces = mesh.faces

    # Shifted copy in the -Z direction
    extruded_vertices = original_vertices.copy()
    extruded_vertices[:, 2] -= extrusion_depth

    # Combine original + extruded vertices
    all_vertices = np.vstack([original_vertices, extruded_vertices])
    num_original = len(original_vertices)

    # Duplicate top faces (original), and bottom faces (flipped)
    top_faces = original_faces
    bottom_faces = original_faces[:, [0, 2, 1]] + num_original

    # Side walls between top and bottom
    side_faces = []
    for face in original_faces:
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            v0_ex = v0 + num_original
            v1_ex = v1 + num_original
            side_faces.append([v0, v1, v1_ex])
            side_faces.append([v0, v1_ex, v0_ex])

    all_faces = np.vstack([top_faces, bottom_faces, side_faces])
    extruded_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)

    return extruded_mesh


def visualize_mesh(mesh):
    mesh.show()

def save_mesh(mesh, path="output/model2.obj"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    print(f"Saved model to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 3D mesh from an image.")
    parser.add_argument("--input_type", type=str, default="image", help="Input type (image)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_file", type=str, default="output/model2.obj", help="Path to save 3D model")
    args = parser.parse_args()

    print("Preprocessing image...")
    img_no_bg = preprocess_image(args.input_file)

    print("Estimating depth...")
    depth_map = estimate_depth(img_no_bg)

    print("Generating 3D mesh...")
    mesh = generate_3d_mesh(depth_map, img_no_bg)

    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}, Watertight: {mesh.is_watertight}")

    mesh = extrude_mesh(mesh)

    print(f"Post-extrusion - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}, Watertight: {mesh.is_watertight}")

    save_mesh(mesh, args.output_file)
    print("Visualizing mesh...")
    visualize_mesh(mesh)