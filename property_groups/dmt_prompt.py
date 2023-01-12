import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty, IntVectorProperty
import os
import sys
from typing import _AnnotatedAlias
from ..generator_process.actions.utils import StepPreviewMode

step_preview_mode_options = [(mode.value, mode.value, '') for mode in StepPreviewMode]

init_image_sources = [
    ('file', 'File', '', 'IMAGE_DATA', 1),
    ('open_editor', 'Open Image', '', 'TPAINT_HLT', 2),
]

# text_to_pc_model = [
#     ('base40M-textvec', 'Text to Mesh', '', 1),
#     ('prompt_to_pc', 'Text to Point Cloud', '', 2),
# ]

image_to_pc_models = [
    ('base40M', '40M model', '', 1),
    ('base300M', '300M model', '', 2),
    ('base1B', '1B model', '', 3),
]

pc_to_mesh_methods = [
    ('openAI', 'openAI', '', 1),
    ('dmtet', 'dmtet', '', 2),
]

run_modes = [
    ('text_to_mesh', 'Text to Mesh', '', 1),
    ('text_to_pc', 'Text to Point Cloud', '', 2),
    ('pc_to_mesh', 'Point Cloud to Mesh', '', 3),
    ('image_to_pc', 'Image to Point Cloud', '', 4),
    ('image_to_mesh', 'Image to Mesh', '', 5),
]

attributes = {
    # Prompt
    #"prompt_structure": EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly"),
    #"use_negative_prompt": BoolProperty(name="Use Negative Prompt", default=False),
    #"negative_prompt": StringProperty(name="Negative Prompt", description="The model will avoid aspects of the negative prompt"),
    "prompt": StringProperty(name="Prompt", description="prompt for text to point cloud", default="A corgi"),

    # Size
    # "width": IntProperty(name="Width", default=512, min=64, step=64),
    # "height": IntProperty(name="Height", default=512, min=64, step=64),

    # Init Image
    "use_init_img": BoolProperty(name="Use Init Image", default=False),
    "init_img_src": EnumProperty(name=" ", items=init_image_sources, default="file"),
    "fit": BoolProperty(name="Fit to width/height", default=True),
    "use_init_img_color": BoolProperty(name="Color Correct", default=True),

    # Advanced
    #"text_to_pc_model": ,
    "image_to_pc_model": EnumProperty(name="Image to PC model", items=image_to_pc_models, default="base1B"),
    "pc_to_mesh_method": EnumProperty(name="PC to mesh method", items=pc_to_mesh_methods, default="openAI"),

    "steps": IntProperty(name="Steps", default=5000, min=1),
    "view_every": IntProperty(name="Mesh rendering frequency", default=500, min=1),
    "step_preview_mode": EnumProperty(name="Step Preview", description="Displays intermediate steps in the 3D Viewport. Disabling can speed up generation", items=step_preview_mode_options, default=1),
    "lr": FloatProperty(name="Mesh learning rate", default=0.001, precision=10),
    "laplacian_weight": FloatProperty(name="Regulization weight", default=0.3, description="Laplacian regulizer weight"),
    "gridres": IntProperty(name="Gridres", default=128),
    "multires": IntProperty(name="Multires", default=4),

    # Run mode
    "run_mode": EnumProperty(name="Run Mode", items=run_modes, default="text_to_mesh")
}

DMTPrompt = type('DMTPrompt', (bpy.types.PropertyGroup,), {
    "bl_label": "DMTPrompt",
    "bl_idname": "dmt_meshes.DMTPrompt",
    "__annotations__": attributes,
})

def generate_args(self):
    args = { key: getattr(self, key) for key in DMTPrompt.__annotations__ }
    args['step_preview_mode'] = StepPreviewMode(args['step_preview_mode'])
    return args

DMTPrompt.generate_args = generate_args