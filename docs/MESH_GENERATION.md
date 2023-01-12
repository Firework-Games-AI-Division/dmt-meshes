# Mesh Generation
1. To open DMT Meshes, go to 3D Viewport
1. Ensure the sidebar is visible by pressing *N* or checking *View* > *Sidebar*
2. Select the *DMT* panel to open the interface

Enter a prompt then click *Generate*. It can take anywhere from a few seconds to a few minutes to generate, depending on your GPU.

## Pipeline
As an overview, the process of DMT Meshes is: text/image -> point cloud -> mesh.

## Prompt

You can provide a text to generate the point cloud/mesh.

## Source image

You can use an image to geneate point cloud/mesh.

## Point cloud

The point cloud generated from the text/image will be shown in this screen. You can select the point cloud to regenerate the mesh with different methods/parameters.

## Advanced

* Image to point cloud model - Pre-trained openAI model to generate mesh from an image.

* Point cloud to mesh model - You can use either point cloud to mesh implementation from openAI or [DMTet](https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dmtet_tutorial.ipynb).

* Steps, mesh rendering, mesh learning rate, regularization, gridres, multires - parameters for point cloud to mesh. Except gridres, other parameters are only for DMTet. We will revamp the UI in the next release. For more information about the parameters, please visit [here](https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dmtet_tutorial.ipynb).

> **NOTE:** DMTet mode only supports gridres=128

## Run mode

You can choose different run mode: text to mesh, image to mesh, text to point cloud, image to point cloud.

# Use with Dream Textures

After generating the mesh, you can use Dream Textures to project texture on it.


