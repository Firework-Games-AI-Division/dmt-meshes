from .actor import Actor

class Generator(Actor):
    """
    The actor used for all background process.
    """
    from .actions.text_to_pc import text_to_pc
    from .actions.text_to_mesh import text_to_mesh
    from .actions.image_to_pc import image_to_pc
    from .actions.pc_to_mesh import pc_to_mesh
    from .actions.image_to_mesh import image_to_mesh
    from .actions.utils import choose_device, get_point_cloud
