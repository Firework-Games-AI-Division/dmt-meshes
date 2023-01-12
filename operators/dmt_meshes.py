import bpy
import hashlib
import numpy as np

from ..generator_process import Generator
from ..generator_process.actions.utils import MeshGenerationResult

from ..preferences import DMTMeshesPreferences, set_pc_list
import uuid

def bpy_mesh(name, verts, faces, step, existing_mesh_object):
    mesh = bpy.data.meshes.new(f'{name}_{step}')
    mesh.from_pydata(verts, [], faces)
    if not existing_mesh_object:
        existing_mesh_object = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(existing_mesh_object)
    else:
        existing_mesh = existing_mesh_object.data
        existing_mesh_object.data = mesh
        bpy.data.meshes.remove(existing_mesh)
    return existing_mesh_object

class DMTMesh(bpy.types.Operator):
    bl_idname = "object.dmt_mesh"
    bl_label = "DMT Mesh"
    bl_description = "Generate a mesh with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
        screen = context.screen
        scene = context.scene
        generated_args = scene.dmt_meshes_prompt.generate_args()
        generated_args['point_cloud_results_selection'] = scene.point_cloud_results_selection
        init_image = None
        if generated_args['use_init_img']:
            match generated_args['init_img_src']:
                case 'file':
                    init_image = scene.init_img
                case 'open_editor':
                    for area in screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            if area.spaces.active.image is not None:
                                init_image = area.spaces.active.image
        if init_image is not None:
            init_image = np.flipud(
                (np.array(init_image.pixels) * 255)
                 .astype(np.uint8)
                 .reshape((init_image.size[1], init_image.size[0], init_image.channels))
            )

        # Setup the progress indicator
        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None
        bpy.types.Scene.dmt_meshes_progress = bpy.props.IntProperty(name="",
                                                                        default=0,
                                                                        min=0,
                                                                        max=generated_args['steps'],
                                                                        update=step_progress_update)
        scene.dmt_meshes_info = "Starting..."

        last_data_block = None
        mesh_name = str(uuid.uuid4())
        def step_callback(_, step_mesh: MeshGenerationResult):
            nonlocal last_data_block
            if step_mesh.final:
                return
            scene.dmt_meshes_progress = step_mesh.step
            if step_mesh.status:
                scene.dmt_meshes_info = step_mesh.status
            if step_mesh.verts is not None and step_mesh.faces is not None:
                last_data_block = bpy_mesh(mesh_name,
                                           step_mesh.verts,
                                           step_mesh.faces,
                                           step_mesh.step,
                                           last_data_block)

        def done_callback(future):
            nonlocal last_data_block
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            mesh_result: MeshGenerationResult | list = future.result()
            set_pc_list('point_cloud_results', Generator.shared().get_point_cloud().result())
            if isinstance(mesh_result, list):
                mesh_result = mesh_result[-1]
            if mesh_result.verts is not None and mesh_result.faces is not None:
                last_data_block = bpy_mesh(mesh_name,
                                        mesh_result.verts,
                                        mesh_result.faces,
                                        mesh_result.step,
                                        last_data_block)
            scene.dmt_meshes_info = ""
            scene.dmt_meshes_progress = 0
            
        def exception_callback(_, exception):
            scene.dmt_meshes_info = ""
            scene.dmt_meshes_progress = 0
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            self.report({'ERROR'}, str(exception))
            raise exception

        gen = Generator.shared()
        def generate_next():
            if generated_args['run_mode'] == 'image_to_mesh' and init_image is not None:
                f = gen.image_to_mesh(init_image=init_image, **generated_args)
            elif generated_args['run_mode'] == 'image_to_pc' and init_image is not None:
                f = gen.image_to_pc(init_image=init_image, **generated_args)
            elif generated_args['run_mode'] == 'text_to_pc':
                f = gen.text_to_pc(**generated_args)
            elif generated_args['run_mode'] == 'text_to_mesh':
                f = gen.text_to_mesh(**generated_args)
            elif generated_args['run_mode'] == 'pc_to_mesh':
                f = gen.pc_to_mesh(pc_idx=generated_args['point_cloud_results_selection'], **generated_args) 

            gen._active_generation_future = f
            f.call_done_on_exception = False
            f.add_response_callback(step_callback)
            f.add_exception_callback(exception_callback)
            f.add_done_callback(done_callback)
        generate_next()
        return {"FINISHED"}

def kill_generator(context=bpy.context):
    Generator.shared_close()
    try:
        context.scene.dmt_meshes_info = ""
        context.scene.dmt_mehses_progress = 0
    except:
        pass

class ReleaseGenerator(bpy.types.Operator):
    bl_idname = "shade.dmt_meshes_release_generator"
    bl_label = "Release Generator"
    bl_description = "Releases the generator class to free up VRAM"
    bl_options = {'REGISTER'}

    def execute(self, context):
        kill_generator(context)
        return {'FINISHED'}

class CancelGenerator(bpy.types.Operator):
    bl_idname = "shade.dmt_meshes_stop_generator"
    bl_label = "Cancel Generator"
    bl_description = "Stops the generator without reloading everything next time"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        gen = Generator.shared()
        return hasattr(gen, "_active_generation_future") and gen._active_generation_future is not None and not gen._active_generation_future.cancelled and not gen._active_generation_future.done

    def execute(self, context):
        gen = Generator.shared()
        gen._active_generation_future.cancel()
        context.scene.dmt_meshes_info = ""
        context.scene.dmt_meshes_progress = 0
        return {'FINISHED'}