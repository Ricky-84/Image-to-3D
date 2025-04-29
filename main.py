import argparse
import numpy as np # type: ignore
from PIL import Image # type: ignore
import torch # type: ignore
from diffusers import StableDiffusionPipeline # type: ignore
from rembg import remove # type: ignore
import trimesh # type: ignore
import pyrender # type: ignore
import matplotlib.pyplot as plt # type: ignore
from transformers import pipeline

# Load MiDaS model for depth estimation
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

def generate_image_from_text(prompt):
    """Generate an image from a text prompt using Stable Diffusion."""
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt + " on a white background").images[0]
    return image

def preprocess_image(image_path=None, text_prompt=None):
    """Preprocess input: load image or generate from text, remove background."""
    if text_prompt:
        print("Generating image from text prompt...")
        img = generate_image_from_text(text_prompt)
    else:
        img = Image.open(image_path).convert("RGB")
    
    # Remove background
    img_no_bg = remove(np.array(img))
    img_no_bg = Image.fromarray(img_no_bg)
    return img_no_bg

def estimate_depth(image):
    """Estimate depth map using MiDaS."""
    depth_map = depth_estimator(image)["depth"]
    depth_array = np.array(depth_map)
    # Normalize depth values to [0, 1]
    depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
    return depth_array

def create_mesh(depth_map, mask, threshold=0.5):
    """Create a 3D mesh from depth map and mask."""
    height, width = depth_map.shape
    vertices = []
    faces = []
    vertex_map = {}

    # Create vertices only for object pixels
    for y in range(height):
        for x in range(width):
            if mask[y, x] > threshold:  # Assuming mask is 0-1 after normalization
                z = depth_map[y, x]
                vertex_idx = len(vertices)
                vertices.append([x, y, z * 100])  # Scale z for visibility
                vertex_map[(x, y)] = vertex_idx

    # Create faces by connecting adjacent object pixels
    for y in range(height - 1):
        for x in range(width - 1):
            if (x, y) in vertex_map and (x + 1, y) in vertex_map and \
               (x, y + 1) in vertex_map and (x + 1, y + 1) in vertex_map:
                v0 = vertex_map[(x, y)]
                v1 = vertex_map[(x + 1, y)]
                v2 = vertex_map[(x, y + 1)]
                v3 = vertex_map[(x + 1, y + 1)]
                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def visualize_mesh(mesh):
    """Visualize the 3D mesh using pyrender."""
    scene = pyrender.Scene()
    mesh_pr = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pr)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def main():
    parser = argparse.ArgumentParser(description="Convert photo or text to 3D model")
    parser.add_argument("--input_type", choices=["image", "text"], required=True,
                        help="Type of input: 'image' or 'text'")
    parser.add_argument("--input_file", help="Path to input image file (for image input)")
    parser.add_argument("--input_text", help="Text prompt (for text input)")
    parser.add_argument("--output_file", default="output.obj",
                        help="Output file path for 3D model (.obj)")
    
    args = parser.parse_args()

    # Validate inputs
    if args.input_type == "image" and not args.input_file:
        raise ValueError("Image input requires --input_file")
    if args.input_type == "text" and not args.input_text:
        raise ValueError("Text input requires --input_text")

    # Preprocess input
    print("Processing input...")
    img_no_bg = preprocess_image(args.input_file if args.input_type == "image" else None,
                                 args.input_text if args.input_type == "text" else None)

    # Convert mask to binary (0 or 1)
    mask = np.array(img_no_bg)[:, :, 3] / 255.0  # Alpha channel from rembg output

    # Estimate depth
    print("Estimating depth...")
    depth_map = estimate_depth(img_no_bg)

    # Create 3D mesh
    print("Generating 3D model...")
    mesh = create_mesh(depth_map, mask)

    # Save mesh
    mesh.export(args.output_file)
    print(f"Model saved to {args.output_file}")

    # Visualize
    print("Visualizing model...")
    visualize_mesh(mesh)

if __name__ == "__main__":
    main()