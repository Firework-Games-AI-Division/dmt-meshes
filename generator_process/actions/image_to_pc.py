def image_to_pc(
               self,
               init_image,
               image_to_pc_model='base1B',
               **kwargs
               ):
    import torch
    import numpy as np
    from uuid import uuid4
    from .utils import MeshGenerationResult
    from tqdm.auto import tqdm
    from pathlib import Path
    from ...absolute_path import absolute_path, DATA_PATH
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config

    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config

    device = self.choose_device()
    img = init_image
    for _, model in MODEL_CONFIGS.items():
        if model['name'] in ['CLIPImagePointDiffusionTransformer',
                             'CLIPImageGridPointDiffusionTransformer',
                             'UpsamplePointDiffusionTransformer',
                             'CLIPImageGridUpsamplePointDiffusionTransformer']:
            model.update({'cache_dir': Path(absolute_path(DATA_PATH))/'pointe_cache'})
    # image to pc
    base_name = image_to_pc_model #'base40M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    status = 'creating upsample model...'
    print(status)
    yield MeshGenerationResult(
    None,
    None,
    0,
    False,
    status
    )
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    status = 'downloading base checkpoint...'
    print(status)
    yield MeshGenerationResult(
    None,
    None,
    0,
    False,
    status
    )
    base_model.load_state_dict(load_checkpoint(base_name, device, cache_dir=Path(absolute_path(DATA_PATH))/'pointe_cache'))

    status = 'downloading upsampler checkpoint...'
    print(status)
    yield MeshGenerationResult(
    None,
    None,
    0,
    False,
    status
    )
    upsampler_model.load_state_dict(load_checkpoint('upsample', device, cache_dir=Path(absolute_path(DATA_PATH))/'pointe_cache'))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]
    pc_path = Path(absolute_path(DATA_PATH))/'pc'
    pc.save(pc_path/f'{uuid4()}.npz')
    yield MeshGenerationResult(
        None,
        None,
        0,
        True
    )
