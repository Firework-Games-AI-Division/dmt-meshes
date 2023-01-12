![DMT Meshes, subtitle: Generative 3D Meshes built-in to Blender](docs/assets/banner.png)

[![Latest Release](https://flat.badgen.net/github/release/Firwork-Games-AI-Division/dmt-meshes)](https://github.com/Firework-Games-AI-Division/dmt-meshes/releases/latest)
[![Total Downloads](https://img.shields.io/github/downloads/Firework-Games-AI-Division/dmt-meshes/total?style=flat-square)](https://github.com/Firework-Games-AI-Division/dmt-meshes/releases/latest)

* Create point clouds or meshes with a simple text prompt or image.

# Installation
Download the [latest release](https://github.com/Firwork-Games-AI-Division/dmt-meshes/releases/latest) and follow the instructions there to get up and running.

## [Setting Up](docs/SETUP.md)
Setup instructions for various platforms and configurations.

## [Mesh Generation](docs/MESH_GENERATION.md)
Create point clouds or meshes with text prompts or images. Learn how to use the various configuration options to get exactly what you're looking for.

# Contributing
After cloning the repository, there a few more steps you need to complete to setup your development environment:
We recommend the [Blender Development](https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development) extension for VS Code for debugging. If you just want to install manually though, you can put the `dmt_meshes` repo folder in Blender's addon directory.
3. After running the local add-on in Blender, setup the model weights like normal.
4. Install dependencies locally
    * Open Blender's preferences window
    * Enable *Interface* > *Display* > *Developer Extras*
    * Then install dependencies for development under *Add-ons* > *DMT Meshes* > *Development Tools*
    * This will download all pip dependencies for the selected platform into `.python_dependencies`

# Credits

Parts of code/scripts are inspired/borrowed from:
 - [dream-textures](https://github.com/carson-katri/dream-textures/)
 - [Point-E](https://github.com/openai/point-e)
 - [DMTet](https://github.com/NVIDIAGameWorks/kaolin/)