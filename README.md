# Image-to-3D:  Generating 3D Models from 2D Image

Project Description

This project, developed as part of the Internsala assignment (Project 6), converts a 2D side-profile image of a toy truck into a 3D mesh model. The script uses depth estimation to infer 3D structure, removes the background for cleaner processing, mirrors the mesh to create a double-sided model, and extrudes it to form a closed, watertight 3D volume. The final output is an .obj file that can be visualized in tools like Blender or MeshLab.

Features
Background Removal: Uses rembg to isolate the toy truck from the background.
Depth Estimation: Employs the Intel/dpt-hybrid-midas model from Hugging Face for depth estimation.
Mesh Generation: Creates a 3D mesh by mapping depth values to vertices and mirroring along the x-axis.
Extrusion and Closure: Manually extrudes the mesh in both z-directions and applies closure techniques to ensure watertightness.
Optimization: Includes downsampling to improve performance and reduce vertex/face counts.
Error Handling: Robust handling of failures in background removal, depth estimation, and mesh processing.

Setup Instructions

Prerequisites
Python: Version 3.8 or higher.
Virtual Environment: Recommended for dependency management.
Hardware: A GPU is recommended for faster depth estimation, but CPU works as well.

Installation
Clone the Repository (if applicable):

git clone [<repository-url>](https://github.com/Ricky-84/Image-to-3D)
cd text-image-to-3D

Create and Activate a Virtual Environment:

python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

Install Dependencies: Create a requirements.txt file with the following content:

torch
numpy
trimesh
Pillow
transformers
rembg

Then install:

pip install -r requirements.txt

Note: If using a GPU, install the appropriate PyTorch version with CUDA support:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Usage
Prepare the Input Image:
Place your side-profile image of the toy truck (e.g., sample_image2.jpg) in the input/ directory.
Ensure the image has a clear background for better removal results.

Run the Script:

python main.py --input_type image --input_file input/sample_image2.jpg --output_file output/model2.obj

--input_type: Set to image (default).
--input_file: Path to your input image.
--output_file: Path to save the generated 3D model (default: output/model2.obj).

Output:
The script will generate a 3D mesh and save it as output/model2.obj.
Debug information (e.g., vertex/face counts, watertightness status) will be printed to the console.
The mesh will attempt to visualize using trimesh.show(). If this fails, open the .obj file in Blender or MeshLab.

Visualize the Result:
Open output/model2.obj in a 3D modeling tool (e.g., Blender, MeshLab).
Check for watertightness using tools like Blender’s “Face Orientation” overlay (blue normals) or “Select Non Manifold Edges”.

Improvements
Background Removal: Utilized rembg for more accurate background removal compared to simple alpha thresholding.
Depth Estimation: Used the Intel/dpt-hybrid-midas model with DPTImageProcessor for depth estimation, ensuring compatibility with future versions of the transformers library.
Mirroring: Mirrored the mesh along the x-axis to create a double-sided model for the side-profile toy truck.
Extrusion: Implemented manual extrusion in both positive and negative z-directions to create a balanced, closed 3D volume.
Closure: Added multiple closure steps (hole filling, normal fixing, and centroid-based boundary closure) to maximize watertightness.
Optimization: Downsampled the input image to reduce vertex/face counts, improving performance (e.g., from 327,968 vertices to ~81,992).
Error Handling: Added robust error handling for background removal, depth estimation, and mesh processing, ensuring the script doesn’t crash on failure.

Limitations
Watertightness: Complex geometries may result in non-watertight meshes, requiring manual repair in a 3D modeling tool like Blender.
Performance: High-resolution images can still lead to large vertex/face counts, slowing down processing. Adjust the downsampling factor in preprocess_image if needed.
Background Removal: rembg may fail on images with complex backgrounds, falling back to a basic alpha threshold method.
Alternative Approach: Considered using Hunyuan3D-1.0 as a backup if the current method fails to produce a closed mesh (not implemented).

Submission Files
input/sample_image2.jpg: Input image of the toy truck.
output/model2.obj: Generated 3D mesh.
requirements.txt: List of dependencies.

README.md: This documentation file.

Future Improvements
Mesh Simplification: Implement a mesh decimation step to further reduce vertex/face counts without losing detail.
Texture Mapping: Add texture mapping to apply the original image colors to the 3D model.
Alternative Models: Explore other depth estimation models or 3D reconstruction tools like Hunyuan3D-1.0 for better results.

Acknowledgments
This project uses the Intel/dpt-hybrid-midas model from Hugging Face for depth estimation.
Background removal is powered by the rembg library.
3D mesh processing relies on the trimesh library.
