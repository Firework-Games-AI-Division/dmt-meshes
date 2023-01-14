![DMT Meshes, subtitle: Generative 3D Meshes built-in to Blender](docs/assets/banner.png)

[![Latest Release](https://flat.badgen.net/github/release/Firework-Games-AI-Division/dmt-meshes)](https://github.com/Firework-Games-AI-Division/dmt-meshes/releases/latest)
[![Join the Discord](https://flat.badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/TsrMtEHe)

* Our goal is to simplfy 3D generation pipeline, this plugin serve to create point clouds or meshes with a simple text prompt or image.

# Installation
To get it up and running, simply download the [latest release](https://github.com/Firework-Games-AI-Division/dmt-meshes/releases/latest), follow the instructions below.

## [Setting Up](docs/SETUP.md)
Outline the setup instructions for various platforms and configurations.

## [Mesh Generation](docs/MESH_GENERATION.md)
Transform your text prompts or images into captivating point clouds and meshes. Discover the various configuration options so that you can craft precisely what you envision!

# instructions
To ensure successful development, you must take a few extra steps after cloning the repository. 
We recommend the [Blender Development](https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development) extension for VS Code for debugging purposes. Alternatively, manual installation is available as well. Simply put the `dmt_meshes` repo folder in Blender's addon directory.
3. After running the local add-on in Blender, setup the model weights like normal.
4. To install dependencies locally, follow these steps: 

    *  Open the Preferences window in Blender
    
    * Check the box to enable *Interface* > *Display* > *Developer Extras*
    
    * Proceed to install dependencies for development under *Add-ons* > *DMT Meshes* > *Development Tools*
    
    * This will download all pip dependencies for the selected platform into `.python_dependencies`

# Contributing to our cause
Our team worked effortlessly to create a usable pipeline to complete/conquer 3D generation. We observed that many people with similar mindsets in this field were attempting to do the same. We wish that by open-sourcing this repository, rather than everyone having to start from scratch, it would serve as a foundation for people to share their discoveries and make improvements on top of what we built, thus accelerating the process.

# Credits

Parts of code/scripts are inspired from:
 - [dream-textures](https://github.com/carson-katri/dream-textures/)
 - [Point-E](https://github.com/openai/point-e)
 - [DMTet](https://github.com/NVIDIAGameWorks/kaolin/)
