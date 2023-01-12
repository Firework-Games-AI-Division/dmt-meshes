def text_to_mesh(
               self,
               prompt,
               text_to_pc_model='base40M-textvec',
               pc_to_mesh_method='dmtet',
               gridres=128,
               lr=1e-3,
               laplacian_weight=0.4,
               steps=5000,
               view_every=500,
               multires=4,
               **kwargs
               ):
    import torch
    import numpy as np
    from uuid import uuid4
    from .utils import MeshGenerationResult, StepPreviewMode
    from tqdm.auto import tqdm
    from pathlib import Path
    from ...absolute_path import absolute_path, DATA_PATH
    import point_e
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config

    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.util.pc_to_mesh import marching_cubes_mesh
    from point_e.util.point_cloud import PointCloud
    from point_e.util.dmtet.dmtet_network import Decoder
    from point_e.util.dmtet.trianglemesh import sample_points
    from point_e.util.dmtet.pointcloud import chamfer_distance
    from point_e.util.dmtet.tetmesh import marching_tetrahedra
    import os

    def laplace_regularizer_const(mesh_verts, mesh_faces):
        term = torch.zeros_like(mesh_verts)
        norm = torch.zeros_like(mesh_verts[..., 0:1])

        v0 = mesh_verts[mesh_faces[:, 0], :]
        v1 = mesh_verts[mesh_faces[:, 1], :]
        v2 = mesh_verts[mesh_faces[:, 2], :]

        term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
        term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
        term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

        two = torch.ones_like(v0) * 2.0
        norm.scatter_add_(0, mesh_faces[:, 0:1], two)
        norm.scatter_add_(0, mesh_faces[:, 1:2], two)
        norm.scatter_add_(0, mesh_faces[:, 2:3], two)

        term = term / torch.clamp(norm, min=1.0)

        return torch.mean(term**2)


    def loss_f(mesh_verts, mesh_faces, points, it):
        pred_points = sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
        chamfer = chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
        if it > steps//2:
            lap = laplace_regularizer_const(mesh_verts, mesh_faces)
            return chamfer + lap * laplacian_weight
        return chamfer

    device = self.choose_device()

    for _, model in MODEL_CONFIGS.items():
        if model['name'] in ['CLIPImagePointDiffusionTransformer',
                             'CLIPImageGridPointDiffusionTransformer',
                             'UpsamplePointDiffusionTransformer',
                             'CLIPImageGridUpsamplePointDiffusionTransformer']:
            model.update({'cache_dir': Path(absolute_path(DATA_PATH))/'pointe_cache'})
    
    # image to pc
    base_name = text_to_pc_model #'base40M' # use base300M or base1B for better results
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
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]
    pc_path = Path(absolute_path(DATA_PATH))/'pc'
    pc.save(pc_path/f'{uuid4()}.npz')
    if pc_to_mesh_method == 'dmtet':
        if isinstance(pc, str):
            pc = PointCloud.load(pc)
        points = pc.coords
        center = (points.max(0)[0] + points.min(0)[0]) / 2
        max_l = (points.max(0)[0] - points.min(0)[0]).max()
        points = ((points - center) / max_l)* 0.9
        points = torch.from_numpy(points).to(device)
        tet_verts = torch.tensor(np.load(os.path.join(point_e.__path__[0], 'util', 'dmtet', 'samples', f'{gridres}_verts.npz'))['data'], dtype=torch.float, device=device)
        tets = torch.tensor(([np.load(os.path.join(point_e.__path__[0], 'util', 'dmtet', 'samples', f'{gridres}_tets_{i}.npz'))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)

        # Initialize model and create optimizer
        model = Decoder(multires=multires).to(device)
        model.pre_train_sphere(1000)

        vars = [p for _, p in model.named_parameters()]
        optimizer = torch.optim.Adam(vars, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

        for i in range(steps):
            pred = model(tet_verts) # predict SDF and per-vertex deformation
            sdf, deform = pred[:, 0], pred[:, 1:]
            verts_deformed = tet_verts + torch.tanh(deform) / gridres # constraint deformation to avoid flipping tets
            mesh_verts, mesh_faces = marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
            mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

            loss = loss_f(mesh_verts, mesh_faces, points, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i) % view_every == 0:
                print('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(i, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
                match kwargs['step_preview_mode']:
                    case StepPreviewMode.NONE:
                        yield MeshGenerationResult(
                            None,
                            None,
                            i,
                            False
                        )
                    case StepPreviewMode.FAST:
                        yield MeshGenerationResult(
                            mesh_verts.detach().cpu().numpy(),
                            mesh_faces.detach().cpu().numpy(),
                            i,
                            False
                        )
            yield MeshGenerationResult(
                mesh_verts.detach().cpu().numpy(),
                mesh_faces.detach().cpu().numpy(),
                i,
                True
            )
    else:
        status = 'creating SDF model...'
        print(status)
        yield MeshGenerationResult(
        None,
        None,
        0,
        False,
        status
        )
        name = 'sdf'
        model = model_from_config(MODEL_CONFIGS[name], device)
        model.eval()

        status = 'loading SDF model...'
        print(status)
        yield MeshGenerationResult(
        None,
        None,
        0,
        False,
        status
        )
        model.load_state_dict(load_checkpoint(name, device, cache_dir=Path(absolute_path(DATA_PATH))/'pointe_cache'))
        
        mesh = marching_cubes_mesh(
            pc=pc,
            model=model,
            batch_size=4096,
            grid_size=gridres, # increase to 128 for resolution used in evals
            progress=True,
        )
        
        yield MeshGenerationResult(
            mesh.verts,
            mesh.faces,
            0,
            True
        )

